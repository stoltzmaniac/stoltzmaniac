import pytest
import numpy as np
import pandas as pd
from stoltzmaniac.data_handler.converter import Converter


def test_converter_class(data_highly_correlated_dataframe):

    with pytest.raises(TypeError):
        Converter()

    converter_df = Converter(input_data=data_highly_correlated_dataframe["dataframe"])
    assert converter_df.input_data.all() is not None
    assert type(converter_df.input_data) == pd.DataFrame
    assert type(converter_df.data) == np.ndarray

    converter_array = Converter(input_data=data_highly_correlated_dataframe["array"])
    assert converter_df.input_data.all() is not None
    assert type(converter_array.input_data) == np.ndarray
    assert type(converter_array.data) == np.ndarray
