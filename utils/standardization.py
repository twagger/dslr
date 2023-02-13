"""Standardization functions"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# nd arrays
import numpy as np
# user modules
from validators import shape_validator, type_validator


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
# normalization functions
@type_validator
@shape_validator({'x': ('m', 'n')})
def normalize_xset(x: np.ndarray) -> np.ndarray:
    """Normalize each feature an entire set of data"""
    try:
        m, n = x.shape
        x_norm = np.empty((m, 0))
        parameters = []
        for feature in range(n):
            x_norm = np.c_[x_norm, z_score(x[:, feature].reshape(-1, 1))]
            parameters.append((np.mean(x), np.std(x)))
        return x_norm, parameters
    except:
        return None


@type_validator
@shape_validator({'x': ('m', 1)})
def z_score(x: np.ndarray) -> np.ndarray:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the
    z-score standardization.
    """
    try:
        z_score_formula = lambda x, std, mean: (x - mean) / std
        zscore_normalize = np.vectorize(z_score_formula)
        x_prime = zscore_normalize(x, np.std(x), np.mean(x))
        return x_prime
    except:
        return None


@type_validator
@shape_validator({'x': ('m', 1)})
def minmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the
        min-max standardization.
    """
    try:
        min_max_formula = lambda x, min, max: (x - min) / (max - min)
        minmax_normalize = np.vectorize(min_max_formula)
        x_prime = minmax_normalize(x, np.min(x), np.max(x))
        return x_prime
    except:
        return None
