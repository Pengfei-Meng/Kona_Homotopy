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
        solver = UserSolver(2,0,1)
        self.km = KonaMemory(solver)
        self.pf = self.km.primal_factory
        self.df = self.km.dual_factory
        self.pf.request_num_vectors(8)
        self.df.request_num_vectors(7)
        optns = {
            'subspace_size' : 50,
            'rel_tol' : 1e-6,
        }
        self.km.allocate_memory()

        self.x = self._generate_vector()
        self.x._primal.equals(1.)
        self.x._dual.equals(1.)

        self.b = self._generate_vector()
        self.b._primal.equals(0.)
        self.b._dual.equals(0.)
        self.b_ = np.array([[0],[0],[0]])
        
        W = np.array([[2, 0],
                         [0, 2]])
        A = np.array([[1,0]])

        W_hat = W
        C_hat = np.array([[1]])

        m = 1
        self.K = np.bmat([[W, A.T], [A, np.zeros((m,m))]])


        self.W = Saddle_Mats(W)
        self.W_hat = Saddle_Mats(W_hat)
        self.A = Saddle_Mats(A)
        self.C_hat = Saddle_Mats(C_hat)

        self.uzawa = UZAWA(self.pf, self.df, self.W, self.W_hat, self.A, self.C_hat, optns)


    def _generate_vector(self):
        design = self.pf.generate()
        dual = self.df.generate()
        return ReducedKKTVector(design, dual)


    def mat_vec(self, in_vec, out_vec):
        in_primal = in_vec._primal._data.data.copy()
        in_dual  = in_vec._dual._data.data.copy()
        out_primal = self.W.dot(in_primal) + self.A.T.dot(in_dual)
        out_dual = self.A.dot(in_primal)

        out_vec._primal._data.data[:] = out_primal[:]
        out_vec._dual._data.data[:] = out_dual[:]


    def test_solve(self):

        # solve the system with UZAWA
        iters, res_p, res_d = self.uzawa.solve(self.b, self.x)
        # calculate expected result
        expected_ = np.linalg.solve(self.K, self.b_)
        # compare actual result to expected
        print 'iters, res_p, res_d', iters, res_p, res_d

        expected = expected_.reshape(len(expected_))
        diff_p = abs(self.x._primal._data.data - expected[:2])
        diff_d = abs(self.x._dual._data.data - expected[-1:])
        diff = np.append(diff_p, diff_d)
        diff = max(diff)
  
        self.assertTrue(diff < 1.e-6)



if __name__ == "__main__":

    unittest.main()
