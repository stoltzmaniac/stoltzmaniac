import pytest
import numpy as np


@pytest.fixture
def DATA_ARRAY_3D() -> np.ndarray:
    my_array = np.array([[1, 12, 2], [2, 3, 4], [3, 9, 6], [4, 1, 8]], dtype=np.float)
    yield my_array
    return print("DATA_ARRAY_3D fixture finished.")


@pytest.fixture
def DATA_ARRAY_2D() -> np.ndarray:
    my_array = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=np.float)
    yield my_array
    return print("DATA_ARRAY_2D fixture finished.")


@pytest.fixture
def DATA_ARRAY_X() -> np.ndarray:
    my_array = np.array([[1., 9.], [2., 13.], [3., 26.], [4., 18.]], dtype=np.float)
    yield my_array
    return print("DATA_ARRAY_2D fixture finished.")


@pytest.fixture
def DATA_ARRAY_REGRESSION_y() -> np.ndarray:
    my_array = np.array([[2.], [4.], [6.], [8.]], dtype=np.float)
    yield my_array
    return print("DATA_ARRAY_REGRESSION_y fixture finished.")


@pytest.fixture
def DATA_ARRAY_CLASSIFICATION_y() -> np.ndarray:
    my_array = np.array([[2], [4], [6], [8]])
    yield my_array
    return print("DATA_ARRAY_CLASSIFICATION_y fixture finished.")
