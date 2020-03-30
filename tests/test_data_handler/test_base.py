import pytest
import numpy as np
import pandas as pd
from stoltzmaniac.data_handler.base import Base


def test_data_handling_base(data_highly_correlated_dataframe):
    """
    Nothing to test yet
    :return:
    """
    with pytest.raises(TypeError):
        Base()

    base_data_df = Base(input_data=data_highly_correlated_dataframe["dataframe"])
    assert base_data_df.input_data.all() is not None
    assert type(base_data_df.input_data) == pd.DataFrame

    base_data_array = Base(input_data=data_highly_correlated_dataframe["array"])
    assert base_data_array.input_data.all() is not None
    assert type(base_data_array.input_data) == np.ndarray
