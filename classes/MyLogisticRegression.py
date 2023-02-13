"""My Logistic Regression module"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# nd arrays
import numpy as np
# for decorators
import inspect
from functools import wraps
# for progress bar
from tqdm import tqdm
# for plot
import matplotlib.pyplot as plt
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from standardization import l2
from validators import shape_validator, type_validator


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
# sigmoid
@type_validator
@shape_validator({'x': ('m', 1)})
def sigmoid_(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid of a vector."""
    try:
        return 1 / (1 + np.exp(-x))
    except:
        return None


# l2 regularizations with decorators
# regulatization of loss
def regularize_loss(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        y, y_hat = args
        m, _ = y.shape
        loss = func(self, y, y_hat)
        regularization_term = (self.lambda_ / (2 * m)) * l2(self.thetas)
        return loss + regularization_term
    return wrapper


# regulatization of gradient
def regularize_grad(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        x, y = args
        m, _ = x.shape
        gradient = func(self, x, y)
        # add regularization
        theta_prime = self.thetas.copy()
        theta_prime[0][0] = 0
        return gradient + (self.lambda_ * theta_prime) / m
    return wrapper


# -----------------------------------------------------------------------------
# MyLogisticRegression class with l2 regularization
# -----------------------------------------------------------------------------
class MyLogisticRegression():
    """My personnal logistic regression to classify things."""

    # We consider l2 penalty only. One may wants to implement other penalties
    supported_penalties = ['l2']

    @type_validator
    @shape_validator({'thetas': ('n', 1)})
    def __init__(self, thetas: np.ndarray, alpha: float = 0.001,
                 max_iter: int = 1000, penalty: str = 'l2',
                 lambda_: float = 1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape((-1, 1))
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalties else 0

    @type_validator
    @shape_validator({'x': ('m', 'n')})
    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from two non-empty
        numpy.ndarray.
        """
        try:
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            return sigmoid_(x_prime.dot(self.thetas))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Calculates the loss by element."""
        try:
            # values check
            min_val = 0
            max_val = 1
            valid_values = np.logical_and(min_val <= y_hat, y_hat <= max_val)
            if not np.all(valid_values):
                print('y / y_hat val must be between 0 and 1', file=sys.stderr)
                return None
            # add a little value to y_hat to avoid log(0) problem
            eps: float=1e-15
            # y_hat = np.clip(y_hat, eps, 1 - eps) < good options for eps
            return -(y * np.log(y_hat + eps) + (1 - y) \
                    * np.log(1 - y_hat + eps))
        except:
            return None

    @type_validator
    @shape_validator({'y': ('m', 1), 'y_hat': ('m', 1)})
    @regularize_loss
    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Compute the logistic loss value."""
        try:
            m, _= y.shape
            return np.sum(self.loss_elem_(y, y_hat)) / m
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
    @regularize_grad
    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the regularized linear gradient of three non-empty
        numpy.ndarray.
        """
        try:
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]
            y_hat = x_prime.dot(self.thetas)
            return x_prime.T.dot(y_hat - y) / m
        except:
            return None

    @type_validator
    @shape_validator({'x': ('m', 'n'), 'y': ('m', 1)})
    def fit_(self, x: np.ndarray, y: np.ndarray, 
             plot: bool = False) -> np.ndarray:
        """
        Fits the model to the training dataset contained in x and y.
        """
        try:
            # calculation of the gradient vector
            m, _ = x.shape
            x_prime = np.c_[np.ones((m, 1)), x]

            # loop (with plot or progress bar)
            if plot is True:
                # gradient update in loop with plot of the learning curve
                fig, axs = plt.figure(1, 1)
                # initialize the line plots
                learn_c, = axs[0].plot([], [], 'r-')
                # label axis
                axs[0].set_xlabel = 'number of iteration'
                axs[0].set_ylabel = 'cost'

                for it, _ in enumerate(tqdm(range(self.max_iter))):
                    # calculate current loss
                    loss = self.loss_(y, self.predict_(x))
                    # Update the lines data
                    learn_c.set_data(np.append(learn_c.get_xdata(), it),
                                     np.append(learn_c.get_ydata(), loss))
                    # Update the axis limits
                    axs[0].relim()
                    axs[0].autoscale_view()

                    # Redraw the figure
                    fig.canvas.draw()

                    # Pause to make animation visible
                    plt.pause(0.001)

                    # calculate the grandient for current thetas
                    gradient = self.gradient_(x, y)
                    # 4. calculate and assign the new thetas
                    self.thetas -= self.alpha * gradient

            else:
                # gradient update in loop with tqdm
                for _ in tqdm(range(self.max_iter)):
                    # calculate the grandient for current thetas
                    gradient = self.gradient_(x, y)
                    # calculate and assign the new thetas
                    self.thetas -= self.alpha * gradient
            
            return self.thetas
        except:
            return None
