import pytest
import pandas as pd
import numpy as np
from stoltzmaniac.utils.convert import (
    pd_dataframe_to_ndarray,
    convert_column_to_numeric,
)


def test_pd_dataframe_to_ndarray():
    # Newly defined DataFrame should match
    df = pd.DataFrame({"foo": ["bar", "is", "silly"], "bar": [1, 2, 3]})
    df_to_array = pd_dataframe_to_ndarray(df)
    assert type(df_to_array) == np.ndarray
    assert df.shape == df_to_array.shape


def test_pd_dataframe_to_ndarray_exception():
    # Newly defined DataFrame should match
    df = {"foo": ["bar", "is", "silly"], "bar": [1, 2, 3]}
    with pytest.raises(TypeError):
        pd_dataframe_to_ndarray(df)


def test_convert_column_to_numeric():
    input_data = np.array([["oh"], ["hi"], [4.3], ["there"], ["me"], ["me"], [6.4]])
    ret = convert_column_to_numeric(input_data)
    comp1 = ret[0] == np.array([4, 2, 0, 5, 3, 3, 1])
    assert comp1.all()
    if not ret[1]:
        assert True
    else:
        assert False

    input_data = np.array([["oh"], ["hi"], ["i"], ["there"], ["me"], ["me"], ["b"]])
    ret = convert_column_to_numeric(input_data)
    comp2 = ret[0] == np.array([4, 1, 2, 5, 3, 3, 0])
    assert comp2.all()
    if not ret[1]:
        assert True
    else:
        assert False

    input_data = np.array([[1], [5], [2], [1], [0], [3], [2]])
    ret = convert_column_to_numeric(input_data)
    comp3 = ret[0] == np.array([[1], [5], [2], [1], [0], [3], [2]])
    assert comp3.all()
    if ret[1]:
        assert True
    else:
        assert False
