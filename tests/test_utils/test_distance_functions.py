import pytest
import numpy as np
from stoltzmaniac.utils.distance_functions import euclidian_distance


def test_euclidean_distance_results():
    a = np.array([[1.0, 2.0, 2.0, 3.0]])
    b = np.array([[2.0, 5.0, 4.0, 1.0]])
    c = euclidian_distance(a, b)
    assert float(c) == pytest.approx(4.242640687119285, 0.01)

def test_euclidean_distance_size_mismatch():
    a = np.array([[1.0, 2.0, 2.0]])
    b = np.array([[2.0, 5.0, 4.0, 1.0]])
    with pytest.raises(ValueError):
        euclidian_distance(a, b)

def test_euclidean_distance_error():
    with pytest.raises(TypeError):
        euclidian_distance(1.0, 2.0)
