import pytest
import numpy as np
from stoltzmaniac.data_handler.utils import splitter


def test_splitter_function_basics(data_highly_correlated_dataframe):
    with pytest.raises(TypeError):
        splitter()

    data_array = splitter(input_data=data_highly_correlated_dataframe["array"])
    assert type(data_array) == dict
    assert type(data_array["test"]) == np.ndarray
    assert type(data_array["train"]) == np.ndarray
    assert (
        data_array["test"].shape[1]
        == data_highly_correlated_dataframe["array"].shape[1]
    )
    assert data_array["test"].shape == (46, 3)
    assert data_array["train"].shape == (107, 3)


def test_splitter_function_input(data_highly_correlated_dataframe):
    with pytest.raises(TypeError):
        splitter(input_data=[1, 2])

    with pytest.raises(ValueError):
        splitter(input_data=data_highly_correlated_dataframe["array"], seed=0.3)

    with pytest.raises(ValueError):
        splitter(
            input_data=data_highly_correlated_dataframe["array"], train_split="blah"
        )

    with pytest.raises(ValueError):
        splitter(input_data=data_highly_correlated_dataframe["array"], train_split=1.9)
