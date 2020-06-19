import pytest
import numpy as np

from stoltzmaniac.utils.distance_functions import euclidian_distance
from stoltzmaniac.models.supervised.k_nearest_neighbors import KNearestNeighbors


def test_knn_euclidean_calculate_distance(DATA_ARRAY_X, DATA_ARRAY_CLASSIFICATION_y, DATA_ARRAY_REGRESSION_y):
    a = np.array([[1.0, 2.0, 2.0, 3.0]])
    b = np.array([[2.0, 5.0, 4.0, 1.0]])
    c = euclidian_distance(a, b)

    # Use any array just to instantiate an instance
    knn = KNearestNeighbors(DATA_ARRAY_X, DATA_ARRAY_CLASSIFICATION_y)
    assert (knn.X == DATA_ARRAY_X).all()

    knn.calculate_distance(a, b, distance_type="euclidean")
    assert float(c) == pytest.approx(4.242640687119285, 0.01)

    # Check error for non specified distance function
    with pytest.raises(TypeError):
        knn.calculate_distance(a, b)

    # Check error for wrong distance_type
    with pytest.raises(ValueError):
        knn.calculate_distance(a, b, distance_type="scott_is_cool")

    with pytest.raises(ValueError):
        KNearestNeighbors(DATA_ARRAY_X, DATA_ARRAY_REGRESSION_y)



def test_knn_with_2_arrays(DATA_ARRAY_3D):
    my_array_x = np.array(
        [
            [1.0, 12.0, 2.0],
            [2.0, 3.0, 4.0],
            [3.0, 9.0, 6.0],
            [1.0, 12.0, 2.0],
            [2.0, 3.0, 4.0],
            [3.0, 9.0, 6.0],
            [1.0, 12.0, 2.0],
            [2.0, 3.0, 4.0],
            [3.0, 9.0, 6.0],
            [1.0, 12.0, 2.0],
            [2.0, 3.0, 4.0],
            [3.0, 9.0, 6.0],
            [1.0, 12.0, 2.0],
            [2.0, 3.0, 4.0],
            [3.0, 9.0, 6.0],
        ]
    )

    my_array_y = np.array([1, 8, 16, 1, 8, 16, 1, 8, 16, 1, 8, 16, 1, 8, 16])

    data_to_predict = np.array(
        [
            [1.0, 12.0, 2.0],
            [2.0, 3.0, 4.0],
            [3.0, 9.0, 6.0],
        ]
    )

    knn = KNearestNeighbors(my_array_x, my_array_y)
    comparison = np.array([1, 8, 16]) == knn.predict(data_to_predict)
    assert comparison.all()

    knn = KNearestNeighbors(my_array_x, my_array_y, scale_type='normalize')
    comparison = np.array([1, 8, 16]) == knn.predict(data_to_predict)
    assert comparison.all()
