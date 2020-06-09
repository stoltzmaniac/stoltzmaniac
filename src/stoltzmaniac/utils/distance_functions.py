import numpy as np


def euclidian_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculates the distance between 2 arrays, similar to a^2 + b^2 = c^2
    Parameters
    ----------
    a: array 1
    b: array 2

    Returns
    -------
    np.ndarray
    """
    if not (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        raise TypeError(
            f"Both input a & b must be of type np.ndarra, currently: a is {type(a)}, b is {type(b)}"
        )
    return np.linalg.norm(a - b)


# TODO: create more distance functions
# def manhattan_distance(a: float, b: float) -> float:
#     pass
#
#
# def chebychev_distance(a: float, b: float) -> float:
#     pass
