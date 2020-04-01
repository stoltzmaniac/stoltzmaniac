import pytest
import numpy as np
from stoltzmaniac.data_handler.utils import converter


def test_converter_function(data_highly_correlated_dataframe):
    with pytest.raises(TypeError):
        converter()

    with pytest.raises(TypeError):
        converter([1, 2, 3])

    data_dataframe = converter(input_data=data_highly_correlated_dataframe["dataframe"])
    assert type(data_dataframe) == np.ndarray
    assert data_dataframe.shape == data_highly_correlated_dataframe["dataframe"].shape

    data_array = converter(input_data=data_highly_correlated_dataframe["array"])
    assert type(data_array) == np.ndarray
    assert data_array.shape == data_highly_correlated_dataframe["array"].shape

    data_array = converter(input_data=np.array(["hi", "there", "yo"]))
    assert data_array.ndim > 1
