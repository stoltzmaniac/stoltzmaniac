import numpy as np


def least_squares(X: np.ndarray, y: np.ndarray, thetas: np.ndarray):
    """
    Calculates the standard cost for X, y matrix and a vector of thetas
    Parameters
    ----------
    X: numpy array of predictor data
    y: numpy array of target variable
    thetas: numpy array of coefficients
    """

    m = len(y)
    predictions = X.dot(thetas)
    return (0.5 * m) * np.sum(np.square(predictions - y))

