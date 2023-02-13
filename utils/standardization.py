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
def normalize_xset(x: np.ndarray, p_means: list = None,
                   p_stds: list = None) -> np.ndarray:
    """Normalize each feature an entire set of data"""
    try:
        m, n = x.shape
        x_norm = np.empty((m, 0))
        means = []
        stds = []
        for feature in range(n):
            serie = x[:, feature].reshape(-1, 1)
            if p_means is not None and p_stds is not None:
                mean = p_means[0][feature][0]
                std = p_stds[0][feature][0]
            else:
                mean = np.mean(serie)
                std = np.std(serie)
            x_norm = np.c_[x_norm, z_score(serie, mean, std)]
            means.append(mean)
            stds.append(std)
        return x_norm, means, stds
    except ValueError as exp:
        print(exp)
        return None


@type_validator
@shape_validator({'x': ('m', 1)})
def z_score(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the
    z-score standardization.
    """
    try:
        z_score_formula = lambda x, mean, std: (x - mean) / std
        zscore_normalize = np.vectorize(z_score_formula)
        x_prime = zscore_normalize(x, mean, std)
        return x_prime
    except:
        return None
