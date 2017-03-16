import numpy as np
import time
import shelve
from mpi4py import MPI
from sumb import SUMB
from kona.user import BaseVector, UserSolver
import math
import pdb, copy, pickle

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a expliclty raised exception.
    """
    def __init__(self, message, comm=None):
        msg = '\n+'+'-'*78+'+'+'\n' + '| SUMB4KONA Error: '
        i = 15
        for word in message.split():
            if len(word) + i + 1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 1 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(78-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        if comm == None:
            print(msg)
        else:
            if comm.rank == 0: print(msg)
        Exception.__init__(self)
               

class SUMB4KONA(UserSolver):
    """
    design vars:   shape, alpha
    """
        
    def __init__(self, objFun, DVCon, conVals, SUMB, outDir='', 
                 debugFlag=False, augFactor=0.0, solveTol=1e-6):
        # store pointers to the problem objects for future use        
        
        self.CFDsolver = SUMB
        self.MBMesh = self.CFDsolver.mesh
        self.aeroProblem = self.CFDsolver.curAP
        self.DVGeo = self.CFDsolver.DVGeo
        self.DVCon = DVCon
        # complain about objective function if it's not valid
        if objFun not in self.CFDsolver.sumbCostFunctions:
            raise Error('__init__(): \
                        Supplied objective function is not known to SUmb!')
        # store objective function strings
        self.objective = objFun
        # objective augmentation factor
        self.augFactor = augFactor
        # linear and adjoint RHS solve tolerance
        self.solveTol = solveTol
        # get geometric design variables
        self.geoDVs = self.DVGeo.getValues()
        # get aerodynamic design variables
        self.aeroDVs = {}

        for key in self.CFDsolver.aeroDVs:
            self.aeroDVs[key] = self.aeroProblem.DVs[key].value
        # merge design variables into single dictionary
        self.designVars = dict(self.geoDVs.items() + self.aeroDVs.items())

        # get dictionary of geometric constraints
        geoCons = {}
        self.DVCon.evalFunctions(geoCons, includeLinear=True, config=None)
        # store the constraint offset values
        self.aeroConVals = {}
        self.geoConVals = {}
        num_eq = 0
        num_ineq = 0

        for key in conVals:
            if key in self.CFDsolver.sumbCostFunctions.keys():
                self.aeroConVals[key] = conVals[key]
                num_ineq += 1

            elif key in geoCons:
                if hasattr(geoCons[key], '__len__'):
                    self.geoConVals[key] = conVals[key]*np.ones(len(geoCons[key]))
                    if key == 'lete': 
                        num_eq += len(geoCons[key])
                    else:
                        num_ineq += len(geoCons[key])

                else:
                    self.geoConVals[key] = conVals[key]
                    if key == 'lete': 
                        num_eq += 1
                    else:
                        num_ineq += 1

            else:
                raise Error('__init__(): \
                            \'%s\' constraint is not valid. \
                            Check problem setup.')

        self.x_min = -0.25*np.ones(len(self.geoDVs['shape']))
        self.x_min = np.insert(self.x_min, 0, 0)

        self.x_max = 0.25*np.ones(len(self.geoDVs['shape']))
        self.x_max = np.insert(self.x_max, 0, 10)        

        # merge aero and geo constraints into single constraint dictionary
        self.constraints = dict(self.geoConVals.items() + self.aeroConVals.items())
            
        # initialize optimization sizes
        self.num_state = self.CFDsolver.getStateSize()  
        # initialize the design norm check to prevent unnecessary mesh warping
        self.cur_design_norm = None

        num_design = 0
        for key in self.designVars:
            if hasattr(self.designVars[key], '__len__'):
                num_design += len(self.designVars[key])
            elif isinstance(self.designVars[key], float):
                num_design += 1
            else:
                raise Error('__init__(): design variables not valid')

        num_sta = self.num_state
        self.local_state = self.CFDsolver.getStates()

        super(SUMB4KONA, self).__init__(num_design = num_design, num_state = num_sta, 
            num_eq = num_eq, num_ineq = num_ineq)
        # set debug informationrm
        self.debug = debugFlag
        if self.get_rank() == 0:
            self.output = True
        else:
            self.output = False
        self.outDir = outDir
        # internal optimization bookkeeping
        self.iterations = 0
        self.totalTime = 0.
        self.startTime = time.clock()
        if self.output:
            file = open(self.outDir+'/kona_timings.dat', 'a')
            file.write('# SUMB4KONA iteration timing history\n')
            titles = '# {0:s}    {1:s}    {2:s}    {3:s}    {4:s}   {5:s}\n'.format(
                'Iter', 'Time (s)', 'Total Time (s)', 'Objective Val', 'Max Constraint', 'dual_norm')
            file.write(titles)
            file.close()
        
    def isNewDesign(self, design_vec): 

        if (np.linalg.norm(design_vec) != self.cur_design_norm):
            return True
        else:
            return False
    
    def enforce_bounds(self, at_design):
        # loop over design variables
        lower_enforced = False
        upper_enforced = False

        for i in xrange(len(at_design)):
                # enforce lower bound
                if at_design[i] < self.x_min[i]:
                    at_design[i] = self.x_min[i]
                    lower_enforced = True
                # enforce upper bound
                if at_design[i] > self.x_max[i]:
                    at_design[i] = self.x_max[i]
                    upper_enforced = True
        if lower_enforced:
            print 'Lower bound enforced!'
        if upper_enforced:
            print 'Upper bound enforced!'

        

    def updateDesign(self, at_design):
        if self.output and self.debug: print('   |-- Updating design ...')

        # put design_vec back into design_dict 
        if isinstance(at_design, np.ndarray):
            design_dict = self.array2dict(at_design)
            self.cur_design_norm = np.linalg.norm(at_design)

        elif isinstance(at_design, dict):
            design_dict = at_design
            self.cur_design_norm = np.linalg.norm(self.dict2array(at_design))
        else:
            raise Error('updateDesign: at_design neither dict or array')

        # pass in GeoDV into the DVGeo object
        self.DVGeo.setDesignVars(design_dict)
        # pass in the AeroDV into aeroProblem object
        self.aeroProblem.setDesignVars(design_dict)
        # propagate changes through SUmb (this warps the mesh)
        self.CFDsolver.setAeroProblem(self.aeroProblem)

    def array2dict(self, in_vec):
        design_dict = copy.deepcopy(self.designVars)
        design_dict['alpha'] = in_vec[0]
        design_dict['shape'] = in_vec[1:]
        return design_dict

    def dict2array(self, in_dict):
        return np.concatenate((np.array([in_dict['alpha']]), in_dict['shape'].flatten()))

################################################################################
#                          kona Toolbox Functions                            #
################################################################################
    

    def get_rank(self):
        return self.CFDsolver.comm.rank

    def allocate_state(self, num_vecs):
        return [BaseVector(len(self.local_state)) \
            for i in range(num_vecs)]        
                
    def eval_obj(self, at_design, at_state):
        if self.output and self.debug: print('>> kona::eval_obj() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design):
            self.updateDesign(at_design)

        self.CFDsolver.setStates(at_state.data)
        cost = 0
        # perform a residual evaluation to propagate the state change
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)

        if math.isnan(resNorm):
            raise Error('eval_obj(): \
                        Residual norm is NaN. kona_state is wrong.')
        # calculate the cost functions and get solution dict
        sol = self.CFDsolver.getSolution()
        Obj = sol[self.objective.lower()] + self.augFactor*np.dot(at_design, at_design)

        return (Obj, cost)
        
    def eval_residual(self, at_design, at_state, store_here):
        if self.output and self.debug: print('>> kona::eval_residual() <<')
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
 
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)

        resNorm = np.linalg.norm(residual)

        if math.isnan(resNorm):
            raise Error('eval_residual(): \
                        Residual norm is NaN. kona_state is wrong.')
        else:
            store_here.data = residual


    def eval_eq_cnstr(self, at_design, at_state):
        if self.output and self.debug: print('>> kona::eval_eq_cnstr() <<')
        # update design;   only lete constraints are equality constraints
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)

        # get and store the geometric constraints  ||  only lete
        geoConsDict = {}
        self.DVCon.evalFunctions(geoConsDict, includeLinear=True, config=None)
        lete_con = geoConsDict['lete'] - self.geoConVals['lete']

        return lete_con

    def eval_ineq_cnstr(self, at_design, at_state):
        if self.output and self.debug: print('>> kona::eval_ineq_cnstr() <<')
        # if the design has changed, update it
        # cl > 0.5    -0.17 < cmy < 0 
        # thick > 0.25*tbase    vol > min_vol
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)

        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem, 
                                              releaseAdjointMemory=False)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('eval_constraints(): \
                        Residual norm is NaN. kona_state is wrong.') 

        # get and store the geometric constraints
        # calculate the cost functions and get solution dict
        aeroFuncsDict = self.CFDsolver.getSolution()

        ineq_cons = {}
        for key in self.aeroConVals:
            ineq_cons[key] = aeroFuncsDict[key.lower()] - self.aeroConVals[key]

        geoConsDict = {}
        self.DVCon.evalFunctions(geoConsDict, includeLinear=True, config=None)

        ineq_cons['thick'] = geoConsDict['thick'] - self.geoConVals['thick']
        ineq_cons['vol'] = geoConsDict['vol'] - self.geoConVals['vol']

        out_vec = np.concatenate((np.array([ineq_cons['cl']]), np.array([ineq_cons['cmy']]), \
            ineq_cons['thick'].flatten(), ineq_cons['vol'].flatten() ))

        return out_vec
                
    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::multiply_dRdX() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update the state variables
        self.CFDsolver.setStates(at_state.data)

        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)

        if math.isnan(resNorm):
            raise Error('multiply_dRdX(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the complete multiplication via SUmb's matrix free routine

        designDot = self.array2dict(in_vec)
        out_vec.data = self.CFDsolver.computeJacobianVectorProductFwd(
            wDot=None, xDvDot=designDot, residualDeriv=True, funcDeriv=False)
        
    def multiply_dRdX_T(self, at_design, at_state, in_vec): 
        if self.output and self.debug: print('>> kona::multiply_dRdX_T() <<')
        
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update the state variables
        self.CFDsolver.setStates(at_state.data)

        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)

        if math.isnan(resNorm):
            raise Error('multiply_dRdX_T(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the complete reverse product via SUmb's matrix free routine
        dRdxProd = self.CFDsolver.computeJacobianVectorProductBwd(
            resBar=in_vec.data, funcsBar=None, 
            wDeriv=False, xDvDeriv=True)
        out_vec = self.dict2array(dRdxProd)

        return out_vec
    
    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec): 
        if self.output and self.debug: print('>> kona::multiply_dRdU() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dRdU(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the multiplication via SUmb matrix-free routines
        out_vec.data = \
        self.CFDsolver.computeJacobianVectorProductFwd(wDot=in_vec.data,
                                                   xDvDot=None,
                                                   residualDeriv=True,
                                                   funcDeriv=False)
        
    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec): 
        if self.output and self.debug: print('>> kona::multiply_dRdU_T() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dRdU_T(): \
                        Residual norm is NaN. kona_state is wrong.')
       
        out_vec.data = \
        self.CFDsolver.computeJacobianVectorProductBwd(resBar=in_vec.data, wDeriv=True)


    def build_precond(self):
        pass
        
    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::apply_precond() <<')
        out_vec.data = self.CFDsolver.globalNKPreCon(
                                  in_vec.data, 
                                  np.zeros(self.num_state))
        # if self.get_rank() == 0:
        #     print 'apply_precond is being called! '

        return 1
        
    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::apply_precond_T() <<')
        out_vec.data = self.CFDsolver.globalAdjointPreCon(
                                  in_vec.data, 
                                  np.zeros(self.num_state))

        # if self.get_rank() == 0:
        #     print 'apply_precond_T is being called! '

        return 1

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCEQdX() <<')
        
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)

        # get the geometric constraint derivatives (these are cheap and small)
        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)

        # 'lete' is the only equality constraints influenced by 'shape'
        DV = 'shape'
        con = 'lete' 
        dCdX = geoSens[con][DV]

        out_vec = np.dot(dCdX, in_vec[1:])
        return out_vec

        
    def multiply_dCEQdU(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCdU() <<')
        return np.zeros( len(self.geoConVals['lete']) )
        
    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCdX_T() <<')

        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)

        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)
        
        DV = 'shape'
        con = 'lete'
        dCdX = geoSens[con][DV]

        out_vec = np.dot(dCdX.transpose(), in_vec)
        return np.insert(out_vec, 0, 0)

    def multiply_dCEQdU_T(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::multiply_dCdU_T() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)

        out_vec.equals_value(0.0)

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdX() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
 
        # cl > 0.5    -0.17 < cmy < 0 
        # thick > 0.25*tbase    vol > min_vol
        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCEQdX(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the dC/dXdv multiplication for aerodynamic constraints
        designDot = self.array2dict(in_vec)
        aeroConProds = \
        self.CFDsolver.computeJacobianVectorProductFwd(wDot=None,
                                                   xDvDot=designDot,
                                                   residualDeriv=False,
                                                   funcDeriv=True)

        ineq_cons = {}
        for con in self.aeroConVals:
            ineq_cons[con] = aeroConProds[con.lower()]   

        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)

        DV = 'shape'
        con1 = 'thick'
        con2 = 'vol'

        dC1dX = geoSens[con1][DV]         
        dC2dX = geoSens[con2][DV] 

        # process the dimension for products
        ineq_cons[con1] = np.dot(dC1dX, in_vec[1:])
        ineq_cons[con2] = np.dot(dC2dX, in_vec[1:])  

        out_vec = np.concatenate((np.array([ineq_cons['cl']]), np.array([ineq_cons['cmy']]), \
                ineq_cons['thick'], ineq_cons['vol'] )) 

        return out_vec


    def multiply_dCINdU(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdU() <<')
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)

        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCdU(): \
                        Residual norm is NaN. kona_state is wrong.')

        funcsDot = \
        self.CFDsolver.computeJacobianVectorProductFwd(wDot=in_vec.data,
                                                   xDvDot=None,
                                                   residualDeriv=False,
                                                   funcDeriv=True)
        ineq_cons = {}
        for con in self.aeroConVals:
            ineq_cons[con] = funcsDot[con.lower()]


        out_vec = np.concatenate((np.array([ineq_cons['cl']]), np.array([ineq_cons['cmy']]), \
             np.zeros(len(self.geoConVals['thick'])), np.array([0.0]) ))

        return out_vec

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdX_T() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)   

        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCdU(): \
                        Residual norm is NaN. kona_state is wrong.')

        aeroConDict = {}      
        aeroConDict['cl'] = in_vec[0]
        aeroConDict['cmy'] = in_vec[1]
  
        funcsDot = self.CFDsolver.computeJacobianVectorProductBwd(
                        resBar=None, funcsBar=aeroConDict,
                        wDeriv=False, xDvDeriv=True)

        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)

        DV = 'shape'
        con1 = 'thick'
        con2 = 'vol'
        dCdX1 = geoSens[con1][DV]        
        dCdX2 = geoSens[con2][DV]  

        invec1 = in_vec[2:2+len(self.geoConVals[con1])]
        invec2 = in_vec[2+len(self.geoConVals[con1]):] 

        out_shape = np.dot(np.transpose(dCdX1), invec1) + np.dot(np.transpose(dCdX2), invec2)
        out_vec = np.insert(out_shape, 0, 0)

        return out_vec + self.dict2array(funcsDot)


    def multiply_dCINdU_T(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdU_T() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)   
        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCdU_T(): \
                        Residual norm is NaN. kona_state is wrong.')
        # separate constraints into aero and geo
        aeroConsDict = {}
        aeroConsDict['cl'] = in_vec[0]
        aeroConsDict['cmy'] = in_vec[1]   

        # perform the {dC/dw}^T multiplication
        out_vec.data = \
        self.CFDsolver.computeJacobianVectorProductBwd(resBar=None,
                                                   funcsBar=aeroConsDict,
                                                   wDeriv=True,
                                                   xDvDeriv=False)

        
    def eval_dFdX(self, at_design, at_state):
        if self.output and self.debug: print('>> kona::eval_dFdX() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update state variables   
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('eval_dFdX(): '
                        'Residual norm is NaN. kona_state is wrong.')
        # get the design variable derivatives with the backward routine
        objFun = {}
        objFun[self.objective] = 1.0
        dJdx = self.CFDsolver.computeJacobianVectorProductBwd(
                        resBar=None, funcsBar=objFun,
                        wDeriv=False, xDvDeriv=True)

        dfdx = self.dict2array(dJdx)

        return dfdx + 2*self.augFactor*at_design
                        
    def eval_dFdU(self, at_design, at_state, store_here):
        if self.output and self.debug: print('>> kona::eval_dFdU() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update state variables
        self.CFDsolver.setStates(at_state.data)
        # import pdb; pdb.set_trace()
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dFdU(): '
                        'Residual norm is NaN. kona_state is wrong.')
        # get the state variable derivatives with the backward routine
        objFun = {}
        objFun[self.objective] = 1.0
        dJdw = self.CFDsolver.computeJacobianVectorProductBwd(
                                  resBar=None, funcsBar=objFun, 
                                  wDeriv=True, xDvDeriv=False)
        store_here.data = dJdw   # *10
        
    def init_design(self):
        # initial design is already set up outside of Kona
        # all we have to do is store the array into Kona memory
        if self.output and self.debug: print('>> kona::init_design() <<')

        init_design = copy.deepcopy(self.designVars)
        # init_design['shape'] = 0.1*np.ones(len(init_design['shape']))

        self.updateDesign(init_design)
        init_vec = self.dict2array(init_design)
        return init_vec

    def init_slack(self):
        at_design = self.init_design()

        # update the design if it changed
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # solve the problem for this design, then solve it

        self.CFDsolver.resetFlow(self.aeroProblem)
        self.CFDsolver(self.aeroProblem, releaseAdjointMemory=False, writeSolution=False)  

        state = self.CFDsolver.getStates()
        at_state = BaseVector(size=len(state), val=state)
        at_slack = self.eval_ineq_cnstr(at_design, at_state)
        return BaseVector(size=len(at_slack), val=at_slack)


    def solve_nonlinear(self, at_design, result):
        if self.output and self.debug: print('>> kona::solve_nonlinear() <<')
        # update the design if it changed
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # solve the problem for this design, then solve it
        if self.output and self.debug:
            print('   |-- Performing system solution ...')
        self.CFDsolver.resetFlow(self.aeroProblem)
        self.CFDsolver(self.aeroProblem, releaseAdjointMemory=False, writeSolution=False)  

        # store the state variables into the Kona memory array
        if self.output and self.debug:
            print('   |-- Saving states into Kona memory ...')
        result.data = self.CFDsolver.getStates()
        if self.output and self.debug:
            print('   |-- Assembling adjoint matrix and solver ...')
        
        # self.CFDsolver.releaseAdjointMemory()

        # check solution convergence and exit
        solveFailed = self.CFDsolver.comm.allreduce(
            self.CFDsolver.sumb.killsignals.fatalfail, op=MPI.LOR)
        if solveFailed:
            return -1
        else:
            return 1
        
    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):   
        if self.output and self.debug: 
            print('>> kona::solve_linear() <<')
            print('   |-- tol: %.4g'%rel_tol)
        
        if not self.CFDsolver.adjointSetup: 
            self.CFDsolver._setupAdjoint()

        globalInner = rhs_vec.inner(rhs_vec)
        globalNorm = np.sqrt(globalInner)

        if globalNorm == 0.0:
            result.data[:] = 0.
            return 1
        else:
            # solve {dR/dw} * kona_state[result] = kona_state[rhs]        
            # self.CFDsolver.sumb.solvedirect(rhs_vec.data, store_here.data, True)
            # print 'rel_tol in solve_linear %f'%rel_tol
            result.data = self.CFDsolver.solveDirectForRHS(rhs_vec.data, relTol=1e-6)
            # check for convergence and return cost metric to Kona
            solveFailed = self.CFDsolver.comm.allreduce(
                self.CFDsolver.sumb.killsignals.adjointfailed, op=MPI.LOR)

            if solveFailed:
                return -1
            else:
                return 1
        
    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        if self.output and self.debug: 
            print('>> kona::solve_adjoint() <<')
            print('   |-- tol: %.4g'%rel_tol)
        # absTol = 1e-6

        if not self.CFDsolver.adjointSetup: 
            self.CFDsolver._setupAdjoint()

        # # calculate initial residual norm       
        globalInner = rhs_vec.inner(rhs_vec)
        globalNorm = np.sqrt(globalInner)

        if globalNorm == 0.0:
            result.data[:] = 0.
            return 1
        else:
            # calculate the relative tolerance we need to pass to 
            self.CFDsolver.sumb.solveadjoint(rhs_vec.data.copy(), result.data, True)

            # check for convergence and return cost metric to Kona
            solveFailed = self.CFDsolver.comm.allreduce(
                self.CFDsolver.sumb.killsignals.adjointfailed, op=MPI.LOR)

            if solveFailed:
                return -1
            else:
                return 1
    

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        if self.output and self.debug: print('>> kona::current_solution() <<')

        # dump solutions
        file_name = self.outDir + '/x_all_%d.pkl'%(num_iter)
        output = open(file_name,'w')
        pickle.dump([curr_design, curr_state, curr_adj, curr_eq, curr_ineq, curr_slack], output)
        output.close()   

        self.CFDsolver.writeSolution(outputDir=self.outDir, 
                                     number=num_iter)

        #-------------------------------------
        geoConsDict = {}
        self.DVCon.evalFunctions(geoConsDict, includeLinear=True, config=None)
        file_name = self.outDir + '/geoCons_%d.pkl'%(num_iter)
        output = open(file_name,'w')
        pickle.dump(geoConsDict, output)  
        output.close()              

        #-------------------------------------
        # time the iteration
        self.endTime = time.clock()
        duration = self.endTime - self.startTime
        self.totalTime += duration
        self.startTime = self.endTime

        objVal, _ = self.eval_obj(curr_design, curr_state)
        min_thick = min(geoConsDict['thick'])

        dual_norm = max(abs(curr_ineq))

        # write timing to file
        timing = '  {0:3d}        {1:4.2f}        {2:4.2f}        {3:4.6g}      {4:4.6f}      {5:4.6f}\n'.format(
            num_iter, duration, self.totalTime, objVal, min_thick, dual_norm)
        if self.output:
            file = open(self.outDir+'/kona_timings.dat', 'a')
            file.write(timing)
            file.close()


        return " "   


    # This part is for matrix-explicit linear cons, approx adjoints nonlinear cons precon
    def total_dCdX_linear(self, at_design, at_state):
        # 
        if self.output and self.debug: print('>> kona::total_dCINdX_linear() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
 
        # thick > 0.25*tbase    vol > min_vol
        # update the state variables
        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)

        DV = 'shape'
        con1 = 'thick'
        con2 = 'vol'
        con3 = 'lete'

        dTdX = geoSens[con1][DV]    
        dTda = np.zeros((len(dTdX), 1))    

        dVdX = geoSens[con2][DV]
        dVda = np.zeros((len(dVdX), 1))   

        dLeTedX = geoSens[con3][DV]
        dLeTeda = np.zeros((len(dLeTedX), 1))

        # Stack together full linear constraint Jacobian, eq, ineq separated

        g_eq = np.hstack([dLeTedX, dLeTeda])
        g_ineq = np.vstack([ np.hstack([dTdX, dTda]),
                             np.hstack([dVdX, dVda]) ])

        return g_eq, g_ineq

    def multiply_dCINdX_nonlinear(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdX_nonlinear() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
 
        # cl > 0.5    -0.17 < cmy < 0 
        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCINdX_nonlinear(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the dC/dXdv multiplication for aerodynamic constraints
        designDot = self.array2dict(in_vec)
        aeroConProds = \
        self.CFDsolver.computeJacobianVectorProductFwd(wDot=None,
                                                   xDvDot=designDot,
                                                   residualDeriv=False,
                                                   funcDeriv=True)
        ineq_cons = {}
        for con in self.aeroConVals:
            ineq_cons[con] = aeroConProds[con.lower()]   

        out_vec = np.concatenate((np.array([ineq_cons['cl']]), np.array([ineq_cons['cmy']]) )) 

        return out_vec

    def multiply_dCINdX_T_nonlinear(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdX_T_nonlinear() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)   

        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCINdX_T_nonlinear(): \
                        Residual norm is NaN. kona_state is wrong.')

        aeroConDict = {}      
        aeroConDict['cl'] = in_vec[0]
        aeroConDict['cmy'] = in_vec[1]
  
        funcsDot = self.CFDsolver.computeJacobianVectorProductBwd(
                        resBar=None, funcsBar=aeroConDict,
                        wDeriv=False, xDvDeriv=True)

        return self.dict2array(funcsDot)

    def multiply_dCINdU_nonlinear(self, at_design, at_state, in_vec):

        if self.output and self.debug: print('>> kona::multiply_dCINdU_nonlinear() <<')
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)

        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCINdU_nonlinear(): \
                        Residual norm is NaN. kona_state is wrong.')

        funcsDot = \
        self.CFDsolver.computeJacobianVectorProductFwd(wDot=in_vec.data,
                                                   xDvDot=None,
                                                   residualDeriv=False,
                                                   funcDeriv=True)
        ineq_cons = {}
        for con in self.aeroConVals:
            ineq_cons[con] = funcsDot[con.lower()]


        out_vec = np.concatenate((np.array([ineq_cons['cl']]), np.array([ineq_cons['cmy']])))

        return out_vec

    def multiply_dCINdU_T_nonlinear(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdU_T_nonlinear() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)   
        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCINdU_T_nonlinear(): \
                        Residual norm is NaN. kona_state is wrong.')
        # separate constraints into aero and geo
        aeroConsDict = {}
        aeroConsDict['cl'] = in_vec[0]
        aeroConsDict['cmy'] = in_vec[1]   

        # perform the {dC/dw}^T multiplication
        out_vec.data = \
        self.CFDsolver.computeJacobianVectorProductBwd(resBar=None,
                                                   funcsBar=aeroConsDict,
                                                   wDeriv=True,
                                                   xDvDeriv=False)
