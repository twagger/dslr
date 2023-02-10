"""
Custom class to perform metrics on a dataset
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
import math
# nd arrays
import numpy as np
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from shape_validator import shape_validator
from type_validator import type_validator


class Metrics():
    """Metrics class"""

    @type_validator
    @shape_validator({'x': ('m', 1)})
    def __init__(self, x: np.ndarray):
        """Constructor"""
        self.x = x[~np.isnan(x)] # filtering nan values
        self.m, self.n = x.shape

    @type_validator
    def mean(self) -> float:
        """Computes the mean of a given non-empty list or array x"""
        result: float = 0
        try:
            return float(np.sum(self.x) / self.m)
        except:
            return None

    @type_validator
    def median(self) -> float:
        """Computes the median of a given non-empty list or array x"""
        return float(self.percentile(50))

    @type_validator
    def percentile(self, p: int) -> float:
        """
        computes the expected percentile of a given non-empty list or array x.
        """
        x_sorted = np.sort(self.x)
        try:
            if p in (0, 100):
                return x_sorted[self.m - 1] if p == 100 else x_sorted[0]
            fractional_rank: float = (p / 100) * (self.m - 1)
            int_part = int(fractional_rank)
            frac_part = fractional_rank % 1
            return (x_sorted[int_part] + frac_part * (x_sorted[int_part + 1]
                    - x_sorted[int_part]))
        except:
            return None

    @type_validator
    def quartiles(self) -> np.ndarray:
        """Computes the 1st and 3rd quartiles of a given non-empty array x"""
        return ([float(self.percentile(25)), float(self.percentile(75))])

    @type_validator
    def var(self) -> float:
        """computes the variance of a given non-empty list or array x"""
        result = 0
        try:
            for num in range(self.m):
                result += (num - self.mean()) ** 2
            return float(result / (self.m - 1))
        except:
            return None

    @type_validator
    def std(self) -> float:
        """
        computes the standard deviation of a given non-empty list or array x
        """
        return math.sqrt(self.var())
