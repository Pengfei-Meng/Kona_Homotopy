import unittest
import numpy as np
import pdb 

from kona.linalg.matrices.preconds import UZAWA
from kona.linalg.matrices.common import IdentityMatrix
from kona.user import UserSolver
from kona.linalg.memory import KonaMemory
from kona.linalg.vectors.composite import ReducedKKTVector

class Saddle_Mats(object):
    def __init__(self, M, transposed=False):
        self.M = M
        self._transposed = transposed

    def product(self, in_vec, out_vec):
        if not self._transposed: 
            out_vec._data.data[:] = np.dot(self.M, in_vec._data.data)
        else:
            out_vec._data.data[:] = np.dot(self.M.T, in_vec._data.data)

    def solve(self, in_vec, out_vec):
        if not self._transposed: 
            out_vec._data.data[:] = np.dot(np.linalg.inv(self.M), in_vec._data.data)
        else:
            out_vec._data.data[:] = np.dot(np.linalg.inv(self.M.T), in_vec._data.data)
  
    @property
    def T(self):
        return self.__class__(self.M, True)


class UZAWASolverTestCase(unittest.TestCase):

    def setUp(self):
        # stokes problem, 1D 
        #  - Delta u + partial P/x = f 
        #  - Div u  = 0 
        # Setting up
        Nx = 50                                # no. of cells
        solver = UserSolver(Nx+1, 0, Nx)
        self.km = KonaMemory(solver)
        self.pf = self.km.primal_factory
        self.df = self.km.dual_factory
        self.pf.request_num_vectors(8)
        self.df.request_num_vectors(7)
        optns = {
            'subspace_size' : 500,
            'rel_tol' : 1e-3,
        }
        self.km.allocate_memory()

        # build stiffness matrix blocks, RHS
        xa = 0.      # interval  x [0,1]
        xb = 1.
        NXT = Nx+1    # no. of grid points
        NPT = Nx     # no. of pressure points, same as cell no.
        dx = (xb - xa)/Nx
        xu = np.linspace(xa, xb, num=NXT)     #[:,np.newaxis]
        xp = xu[:-1] + dx/2

        self.W = 2*np.eye(NXT, k=0) - np.eye(NXT, k=1) - np.eye(NXT, k=-1)
        self.W = self.W/dx

        self.A = np.eye(NPT,NXT,k=0) - np.eye(NPT,NXT, k=1)
        self.O = np.zeros((NPT, NPT))

        self.rhs = self._generate_vector()
        self.X = self._generate_vector()

        # self.rhs_f = self.pf.generate()
        # self.rhs_g = self.df.generate()

        self.rhs._primal._data.data = self._getQ_f(xu)*dx
        self.rhs._dual._data.data = self._getQ_g(xp)*dx

        # self.x = self.pf.generate()
        # self.y = self.df.generate()

        self.X.equals(0.)

        # pdb.set_trace()
        self.W_hat = np.diag(np.diag(self.W))
        self.C_hat = np.eye(NPT)

        self.K = np.bmat([[self.W, self.A.transpose()], [self.A, self.O]])
        self.b_ = np.concatenate((self.rhs._primal._data.data, self.rhs._dual._data.data), axis=0)

        self.uzawa = UZAWA([self.pf, self.df], Saddle_Mats(self.W), Saddle_Mats(self.W_hat), 
            Saddle_Mats(self.A), Saddle_Mats(self.C_hat), optns)


    def _generate_vector(self):
        design = self.pf.generate()
        dual = self.df.generate()
        return ReducedKKTVector(design, dual)

    def _getEX(self, x, iOption=1):
        if iOption == 1:
            u = 1. + x + pow(x,2)
            return u
        else:
            raise NotImplementedError

    def _getEX_x(self, x, iOption=1):
        if iOption == 1:
            u_x = 1. + 2.*x
            p_x = 1.*np.ones_like(x)
            return u_x, p_x
        else:
            raise NotImplementedError

    def _getEX_xx(self, x, iOption=1):
        if iOption == 1:
            u_xx = 2.*np.ones_like(x)
            return u_xx
        else:
            raise NotImplementedError

    def _getQ_f(self, x, iOption=1):
        u_xx = self._getEX_xx(x, iOption)
        u_x, p_x = self._getEX_x(x, iOption)
        return -u_xx + p_x

    def _getQ_g(self, x, iOption=1):
        u_x, p_x = self._getEX_x(x, iOption)
        return -u_x


    def test_solve(self):

        # solve the system with UZAWA

        Converged, iters, norm_total = self.uzawa.solve(self.rhs, self.X)

        # calculate expected result
        expected_ = np.linalg.solve(self.K, self.b_)
        # compare actual result to expected
        print 'iters, final residual norm: ', iters, norm_total

        expected = expected_.reshape(len(expected_))
        diff_p = abs(self.X._primal._data.data - expected[ : self.W.shape[0] ])
        diff_d = abs(self.X._dual._data.data - expected[ -self.A.shape[0] : ])
        diff = np.append(diff_p, diff_d)
        diff = max(diff)
  
        self.assertTrue(diff < 1.e-3)



if __name__ == "__main__":

    unittest.main()
