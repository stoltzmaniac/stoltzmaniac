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
    my_array = np.array(
        [[1.0, 9.0], [2.0, 13.0], [3.0, 26.0], [4.0, 18.0]], dtype=np.float
    )
    yield my_array
    return print("DATA_ARRAY_2D fixture finished.")


@pytest.fixture
def DATA_ARRAY_REGRESSION_y() -> np.ndarray:
    my_array = np.array([[2.0], [4.0], [6.0], [8.0]], dtype=np.float)
    yield my_array
    return print("DATA_ARRAY_REGRESSION_y fixture finished.")


@pytest.fixture
def DATA_ARRAY_CLASSIFICATION_y() -> np.ndarray:
    my_array = np.array([[2], [4], [6], [8]])
    yield my_array
    return print("DATA_ARRAY_CLASSIFICATION_y fixture finished.")
