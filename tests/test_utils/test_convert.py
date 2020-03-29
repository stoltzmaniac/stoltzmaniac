import pytest
import pandas as pd
import numpy as np
from stoltzmaniac.utils.convert import pd_dataframe_to_ndarray


def test_pd_dataframe_to_ndarray():
    # Newly defined DataFrame should match
    df = pd.DataFrame({"foo": ["bar", "is", "silly"], "bar": [1, 2, 3]})
    df_to_array = pd_dataframe_to_ndarray(df)
    assert type(df_to_array) == np.ndarray
    assert df.shape == df_to_array.shape
