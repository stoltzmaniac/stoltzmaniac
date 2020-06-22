import numpy as np


def add_intercept(data: np.ndarray):
    """
    Adds a column of np.ones if an intercept is desired
    Parameters
    ----------
    data: array of X predictor variables
    """
    return np.c_[data, np.ones(len(data))]