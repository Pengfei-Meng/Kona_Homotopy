import numpy as np 

from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.matrices.hessian.basic import BaseHessian, QuasiNewtonApprox
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.hessian import LagrangianHessian

from kona.options import get_opt

class UZAWA(BaseHessian):
    """
    Inequlaity only case:

    This object is a standalone solver for solving the Saddle Point system. 
    The saddle point system is :

    .. math::
        \\begin{bmatrix}
        \\mathcal{W} && \mathcal{A}^T \\\\
         \mathcal{A} && 0  
        \\end{bmatrix}
        \\begin{bmatrix}
        \\Delta x \\\\
        \\Delta \\lambda
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        \\f  \\\\
        \\g 
        \\end{bmatrix}
    """

    def __init__(self, vector_factories, optns={}):

        super(UZAWA, self).__init__(vector_factories, None)
        
        self.primal_factory.request_num_vectors(50)
        self.state_factory.request_num_vectors(2)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(50)

        self.rel_tol = get_opt(optns, 1e-2, 'rel_tol')
        self.abs_tol = get_opt(optns, 1e-3, 'abs_tol')        
        self.max_iter = get_opt(optns, 20, 'max_iter')  
        self.max_stored = get_opt(optns, 10, 'max_stored')  
        print 'max_iter inside UZAWA', self.max_iter
        self.W = LagrangianHessian( vector_factories )
        self.A = TotalConstraintJacobian( vector_factories )

        bfgs_optns = {'max_stored': self.max_stored}
        self.W_hat = LimitedMemoryBFGS(self.primal_factory, bfgs_optns)  
        self.C_hat = LimitedMemoryBFGS(self.ineq_factory, bfgs_optns) 

        self.iter_count = 0
        self._allocated = False


    def _generate_primal(self):
        if self.ineq_factory is None:
            return self.primal_factory.generate()
        else:
            prim = self.primal_factory.generate()
            dual_ineq = self.ineq_factory.generate()        
            return CompositePrimalVector(prim, dual_ineq)

    def _generate_dual(self):
        if self.ineq_factory is not None:
            if self.eq_factory is not None:
                dual_eq = self.eq_factory.generate()
                dual_ineq = self.ineq_factory.generate()
                out = CompositeDualVector(dual_eq, dual_ineq)
            else:    
                out = self.ineq_factory.generate()
        else:
            out = self.eq_factory.generate()

        return  out

    def _generate_kkt(self):
        primal = self._generate_primal()
        dual = self._generate_dual()
        return ReducedKKTVector(primal, dual)

    def linearize(self, X, state, adjoint, mu, dLdX, dLdX_oldual, dLdX_oldprimal):

        self.at_slack_data = X.primal.slack.base.data
        if self.eq_factory is not None:
            self.at_dual_eq_data = X.dual.eq.base.data
            self.at_dual_ineq_data = X.dual.ineq.base.data
        else:
            self.at_dual_ineq_data = X.dual.base.data


        self.iter_count += 1
        self.W.linearize(X, state, adjoint)
        self.A.linearize(X.primal.design, state)

        if self.iter_count == 1:
            if not self._allocated:
                self.dldx_old = self._generate_kkt()
                self.dldx = self._generate_kkt()
                self.x_old = self._generate_kkt()
                self.c_rhs = self._generate_dual()
                self.c_rhs_old = self._generate_dual()
                
                self.c_work = self._generate_primal()

                self.R_p = self._generate_primal()
                self.R_work = self._generate_primal()
                self.Z_p = self._generate_primal()
        
                self.Wz = self._generate_primal()
                self.Az = self._generate_primal()
                self.What_Az = self._generate_primal()

                self.R_d = self._generate_dual()
                self.Z_d = self._generate_dual()

                self.design_work = self.primal_factory.generate()
                self.slack_work = self.ineq_factory.generate()

                self._allocated = True

            self.W_hat.norm_init = 1.0   
            self.C_hat.norm_init = 1.0             

        else:       
            self.x_old.minus(X)
            self.x_old.times(-1.0)

            self.dldx_old.minus(dLdX_oldual)
            self.dldx_old.times(-1.0)
            # self.dldx.equals(dLdX)
            # self.dldx.minus(dLdX_oldual)
            self.W_hat.add_correction(self.x_old.primal.design, self.dldx.primal.design)

            # C_hat approximation, AW^(-1)f - g || suppose primal, dual, k, k+1
            self.W_hat_solve(dLdX_oldprimal.primal, self.c_work)
            # self.A.product(self.c_work, self.c_rhs)
            self.Ag_I_product(self.c_work, self.c_rhs)  
            self.c_rhs.minus(dLdX_oldprimal.dual)
            self.c_rhs_old.minus(self.c_rhs)
            self.c_rhs_old.times(-1.0)
            self.C_hat.add_correction(self.x_old.dual, self.c_rhs_old)

            # #import pdb; pdb.set_trace()
            # self.W_hat_solve(dLdX.primal, self.c_work)
            # self.Ag_I_product(self.c_work, self.c_rhs)
            # self.c_rhs.minus(dLdX.dual)

            # self.W_hat_solve(dLdX_oldprimal.primal, self.c_work)
            # self.Ag_I_product(self.c_work, self.c_rhs_old)
            # self.c_rhs_old.minus(dLdX_oldprimal.dual)

            # self.c_rhs.minus(self.c_rhs_old)
            # self.C_hat.add_correction(self.x_old.dual, self.c_rhs)

        # record x_old and dfdx_old
        self.x_old.equals(X)
        self.dldx_old.equals(dLdX)

        # rhs for the C_hat BFGS secant equation
        self.W_hat_solve(dLdX.primal, self.c_work)
        self.Ag_I_product(self.c_work, self.c_rhs_old)
        self.c_rhs_old.minus(dLdX.dual)
    

    def W_Lam_product(self, in_vec, out_vec):
        # [ W, 0   ]  [in_vec.design] = [ W * in_design   ]
        # [ 0, -Lam ] [in_vec.slack ] = [ -Lam * in_slack ]
        in_design = in_vec.design
        in_slack = in_vec.slack 
        out_design = out_vec.design
        out_slack = out_vec.slack

        self.W.product(in_vec, out_vec)
        out_slack.base.data = - self.at_dual_ineq_data * in_slack.base.data

    def AgT_S_product(self, in_vec, out_vec):
        # [ Ag^T ]             [ out_design ]
        # [ -S   ]  in_dual =  [ out_slack  ]
        in_dual = in_vec
        out_design = out_vec.design
        out_slack = out_vec.slack

        self.A.T.product(in_dual, out_design)
        out_slack.base.data = -self.at_slack_data*in_dual.base.data

    def Ag_I_product(self, in_vec, out_vec):
        # [ Ag  -I ] [ in_design ]  = out_dual 
        #            [ in_slack  ]   
        in_design = in_vec.design
        in_slack = in_vec.slack 
        out_dual = out_vec

        self.A.product(in_design, out_dual)
        out_dual.minus(in_slack)

    def W_hat_solve(self, in_vec, out_vec):
        rhs_design = in_vec.design
        rhs_slack = in_vec.slack 
        out_design = out_vec.design
        out_slack = out_vec.slack

        self.W_hat.solve(rhs_design, out_design)
        out_slack.base.data = -1.0/self.at_dual_ineq_data * rhs_slack.base.data

        # import pdb; pdb.set_trace()

    def solve(self, in_vec, out_vec):  
        # in_vec is RHS vector for the Sadddle-point system
        # out_vec is the solution, or preconditioned in_vec
        Converged = False

        rhs_f = in_vec.primal
        rhs_g = in_vec.dual
        x = out_vec.primal
        y = out_vec.dual


        iters = 0
        for i in xrange(self.max_iter):

            self.W_Lam_product(x, self.R_p)

            self.AgT_S_product(y, self.R_work) 

            self.R_p.plus(self.R_work)
            self.R_p.times(-1.)
            self.R_p.plus(rhs_f)                    # residual   fi 

            beta_p = self.R_p.norm2
            self.W_hat_solve(self.R_p, self.Z_p)    # preconditioned residual ri
            
            # omega
            if (self.R_p.norm2 == 0.): 
                omega = 1.
            else: 
                self.W_Lam_product(self.Z_p, self.Wz)
                zWz = self.Z_p.inner(self.Wz)
                omega = self.R_p.inner(self.Z_p) / zWz


            self.Z_p.times(omega)
            x.plus(self.Z_p)
           
            self.Ag_I_product(x, self.R_d)            
            self.R_d.minus(rhs_g)                    # residual   gi

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.C_hat.solve(self.R_d, self.Z_d)     # preconditioned residual di
            # self.Z_d.equals(self.R_d)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            beta_d = self.R_d.norm2

            if (self.R_d.norm2 == 0.):
                tau = 1.
            else: 
                self.AgT_S_product(self.Z_d, self.Az)
                self.W_hat_solve(self.Az, self.What_Az) 
                tau = self.R_d.inner(self.Z_d) / self.What_Az.inner(self.Az)

            if omega <= 1:
                theta = (1. - np.sqrt(1-omega))/2 
            else:
                theta = 0.25*omega

            self.Z_d.times(theta*tau)
            y.plus(self.Z_d)

            norm_total = np.sqrt( pow(beta_p,2) + pow(beta_d, 2) )

            if (i==0):
                norm0 = norm_total
            elif (norm_total < norm0*self.rel_tol):
                Converged = True
                print 'UZAWA CONVERGED!!'
                break

            iters += 1

        print 'norm0, norm_total: ', norm0, norm_total
        import pdb; pdb.set_trace()
        return Converged, iters, norm_total
