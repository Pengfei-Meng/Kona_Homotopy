import numpy as np 

from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.matrices.hessian.basic import BaseHessian, QuasiNewtonApprox
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
from kona.linalg.vectors.composite import ReducedKKTVector

from kona.options import get_opt

class UZAWA(BaseHessian):
    """
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

        super(UZAWA, self).__init__(vector_factories, optns)
        from kona.linalg.matrices.hessian import LagrangianHessian
        # get vector factories
        self.primal_factory = None
        # self.state_factory = None
        self.dual_factory = None

        for factory in self.vec_fac:
            if factory._vec_type is PrimalVector:
                self.primal_factory = factory
            # elif factory._vec_type is StateVector:
            #     self.state_factory = factory
            elif factory._vec_type is DualVector:
                self.dual_factory = factory

        self.rel_tol = get_opt(optns, 1e-2, 'rel_tol')
        self.abs_tol = get_opt(optns, 1e-3, 'abs_tol')        
        self.max_iter = get_opt(optns, 4, 'max_iter')  
        self.max_stored = get_opt(optns, 4, 'max_stored')  
        print 'max_iter inside UZAWA', self.max_iter
        self.W = LagrangianHessian( vector_factories )
        self.A = TotalConstraintJacobian( vector_factories )

        bfgs_optns = {'max_stored': self.max_stored}
        self.W_hat = LimitedMemoryBFGS(vector_factories, bfgs_optns)  
        self.C_hat = LimitedMemoryBFGS(vector_factories, bfgs_optns) 

        self.iter_count = 0
        self._allocated = False

    def _generate_primal_vector(self):
        design = self.primal_factory.generate()
        slack = self.dual_factory.generate()
        return CompositePrimalVector(design, slack)

    def _generate_KKT_vector(self):
        design = self.primal_factory.generate()
        slack = self.dual_factory.generate()
        primal = CompositePrimalVector(design, slack)
        dual = self.dual_factory.generate()
        return ReducedKKTVector(primal, dual)


    def linearize(self, X, state, adjoint, dLdX, dLdX_oldual):

        self.iter_count += 1
        self.W.linearize(X, state, adjoint)
        self.A.linearize(X, state)

        if self.iter_count == 1:
            if not self._allocated:
                self.primal_factory.request_num_vectors(9)
                self.dual_factory.request_num_vectors(6+6+2)

                self.dldx_old = self._generate_KKT_vector()
                self.dldx = self._generate_KKT_vector()
                self.x_old = self._generate_KKT_vector()
                self.c_rhs = self.dual_factory.generate()
                self.c_rhs_old = self.dual_factory.generate()
                
                self.c_work = self._generate_primal_vector()

                self.R_p = self._generate_primal_vector()
                self.R_work = self._generate_primal_vector()
                self.Z_p = self._generate_primal_vector()
        
                self.Wz = self._generate_primal_vector()
                self.Az = self._generate_primal_vector()
                self.What_Az = self._generate_primal_vector()

                self.R_d = self.dual_factory.generate()
                self.Z_d = self.dual_factory.generate()

                self._allocated = True

            self.W_hat.norm_init = 1.0   
            self.C_hat.norm_init = 1.0             

        else:       
            self.x_old.minus(X)
            self.x_old.times(-1.0)

            self.dldx.equals(dLdX_oldual)
            self.dldx_old.minus(self.dldx)
            self.dldx_old.times(-1.0)
            self.W_hat.add_correction(self.x_old._primal, self.dldx_old._primal)

            # C_hat approximation, AW^(-1)f - g || suppose primal, dual, k, k+1
            self.W_hat.solve(dLdX_oldual._primal, self.c_work)
            # self.A.product(self.c_work, self.c_rhs)
            self.A.approx.product(self.c_work, self.c_rhs)  
            self.c_rhs.minus(dLdX_oldual._dual)
            self.c_rhs_old.minus(self.c_rhs)
            self.c_rhs_old.times(-1.0)

            self.C_hat.add_correction(self.x_old._dual, self.c_rhs_old)
       

        # record x_old and dfdx_old
        self.x_old.equals(X)
        self.dldx_old.equals(dLdX)

        # rhs for the C_hat BFGS secant equation
        self.W_hat.solve(dLdX._primal, self.c_work)
        self.A.product(self.c_work, self.c_rhs_old)
        self.c_rhs_old.minus(dLdX._dual)
    


    def solve(self, in_vec, out_vec):  
        # in_vec is RHS vector for the Sadddle-point system
        # out_vec is the solution, or preconditioned in_vec
        Converged = False

        b_f = in_vec._primal
        b_g = in_vec._dual
        x = out_vec._primal
        y = out_vec._dual


        iters = 0
        for i in xrange(self.max_iter):

            self.W.product(x, self.R_p)

            self.A.T.product(y, self.R_work) 

            self.R_p.plus(self.R_work)
            self.R_p.times(-1.)
            self.R_p.plus(b_f)

            beta_p = self.R_p.norm2
            self.W_hat.solve(self.R_p, self.Z_p)
            
            # omega
            if (self.R_p.norm2 == 0.): 
                omega = 1.
            else: 
                self.W.product(self.Z_p, self.Wz)
                zWz = self.Z_p.inner(self.Wz)
                omega = self.R_p.inner(self.Z_p) / zWz


            self.Z_p.times(omega)
            x.plus(self.Z_p)
           
            self.A.product(x, self.R_d)            
            self.R_d.minus(b_g) 

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.C_hat.solve(self.R_d, self.Z_d) 
            # self.Z_d.equals(self.R_d)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            beta_d = self.R_d.norm2

            if (self.R_d.norm2 == 0.):
                tau = 1.
            else: 
                self.A.T.product(self.Z_d, self.Az)
                self.W_hat.solve(self.Az, self.What_Az) 
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
                break

            iters += 1


        return Converged, iters, norm_total
