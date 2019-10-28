"""
Derived module from dmdbase.py for sparsity-promoting dmd.

Reference:
- [1] M. R. Jovanovi´cjovanovi´c, P. J. Schmid, and J. W. Nichols, “Sparsity-promoting dynamic mode decomposition,”
Phys. FLUIDS, vol. 26, p. 24103, 2014.
- [2] https://github.com/aaren/sparse_dmd
"""
from __future__ import division
from builtins import range
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from .dmdbase import DMDBase


class SparseDMD(object):
    """
    Sparsity-promoting Dynamic Mode Decomposition. It is a post-processing procedure,
    so it relies on an already computed DMD.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.
    """

    def __init__(self,
                 # svd_rank=0,
                 # tlsq_rank=0,
                 # exact=False,
                 # opt=False,
                 penalty=1e-1,
                 eps_abs=1e-6,
                 eps_rel=1e-4,
                 maxiter=100000,
                 ):
        # super(SparseDMD, self).__init__(svd_rank, tlsq_rank, exact, opt)

        # initialize sparse hyperparameters
        self.penalty = penalty
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.maxiter = maxiter

    @property
    def sparse_parameters(self):
        model_params = {
            "l1_penalty": self.penalty,
            "eps_abs": self.eps_abs,
            "eps_rel": self.eps_rel,
            "maxiter": self.maxiter,
        }
        return model_params

    def fit(self, dmd_obj, gamma=None):
        """
        Computes the sparse DMD given a precomputed DMD.

        Parameters
        ----------
        dmd_obj : (DMDBase) a precomputed DMD

        Returns
        -------

        """
        # check if passed in dynamic mode decomposition fits
        if dmd_obj._Atilde is None:
            raise RuntimeError(f"User should pass in a DMD object that has already"
                               f"fitted to data. Sparse DMD is a post-processing step.")

        # create range of sparsity parameters
        if gamma is None:
            gamma = np.logspace(-2, 6, 100)
        self.xpol = None
        self.Nz = None

        return self

    def _fix_sparsity_structure(self):
        """
        Private method to fix the sparsity structure solved from l1-regularization.
        Adjusts the values of the DMD amplitudes to optimally approximate the data
        sequence snapshots passed into DMD.

        Implements a constrained optimization approach:

        `min_{\alpha} J(\alpha)` subject to `E^T \alpha = 0`

        Returns
        -------

        """
        pass

    def _solve_admm(self, z, y, gamma):
        """
        Private method to implement Alternating Direction of Multipliers.
        This will solve the l1-regularized optimization problem to determine
        the non-zero amplitudes to keep within DMD.

        Implements an unconstrained optimization approach:

        `min_{\alpha} J(\alpha) + \gamma \sum_{i=1}^r |\alpha_i|`

        Parameters
        ----------
        z :
        y :
        gamma :

        Returns
        -------

        """
        pass
