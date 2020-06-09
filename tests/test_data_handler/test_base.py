import pytest
import numpy as np
from stoltzmaniac.data_handler.base import ArrayData


def test_array_data_input_types(DATA_ARRAY_3D):


    # Test data which cannot be converted to np.ndarray using dtype=np.float
    with pytest.raises(ValueError):
        ArrayData({"a": 1, "b": 3})

    with pytest.raises(ValueError):
        ArrayData({1, 5, 3, 2})

    array_data = ArrayData(DATA_ARRAY_3D)
    assert (array_data.raw_data == DATA_ARRAY_3D).all()

    array_data = ArrayData([1, 2, 4])
    assert (array_data.raw_data == np.array([[1.0, 2.0, 4.0]])).all()


def test_array_data_description(DATA_ARRAY_3D):
    array_data = ArrayData(DATA_ARRAY_3D)

    exp = (
        f"Data specs:\n"
        f"type: {type(array_data.raw_data)}\n"
        f"shape: {array_data.raw_data.shape}\n"
        f"ndim: {array_data.raw_data.ndim}"
    )

    assert print(exp) == print(array_data)


