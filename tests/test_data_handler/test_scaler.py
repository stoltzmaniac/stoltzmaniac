import pytest
import numpy as np
from stoltzmaniac.data_handler.utils import scaler


def test_scaler_function_basics(data_highly_correlated_dataframe):
    # TODO: Find a decent way to test scale_type for ["normalize", "standardize", "min_max", "scale"]
    with pytest.raises(TypeError):
        scaler()

    data_array = scaler(input_data=data_highly_correlated_dataframe["array"])
    assert type(data_array) == np.ndarray
    assert data_array.shape == data_highly_correlated_dataframe["array"].shape


def test_scaler_function_input(data_highly_correlated_dataframe):
    with pytest.raises(ValueError):
        scaler(input_data=data_highly_correlated_dataframe["array"], scale_type="blah")


def test_scaler_function_none(data_highly_correlated_dataframe):
    ret = scaler(input_data=data_highly_correlated_dataframe["array"])
    comparison = ret == data_highly_correlated_dataframe["array"]
    assert comparison.all()

    ret = scaler(input_data=data_highly_correlated_dataframe["array"], scale_type=None)
    comparison = ret == data_highly_correlated_dataframe["array"]
    assert comparison.all()

    with pytest.raises(ValueError):
        scaler(input_data=np.ndarray([]))


def test_scaler_function_min_max(data_highly_correlated_dataframe):
    input_data = data_highly_correlated_dataframe["array"]
    ret = scaler(input_data=input_data, scale_type="min_max")
    calcs = (input_data - np.min(input_data, axis=0)) / (
        np.max(input_data, axis=0) - np.mean(input_data, axis=0)
    )
    comparison = ret == calcs
    assert comparison.all()


def test_scaler_function_normalize(data_highly_correlated_dataframe):
    input_data = data_highly_correlated_dataframe["array"]
    ret = scaler(input_data=input_data, scale_type="normalize")
    calcs = (input_data - np.mean(input_data, axis=0)) / (
        np.max(input_data, axis=0) - np.min(input_data, axis=0)
    )
    comparison = ret == calcs
    assert comparison.all()


def test_scaler_function_standardize(data_highly_correlated_dataframe):
    input_data = data_highly_correlated_dataframe["array"]
    ret = scaler(input_data=input_data, scale_type="standardize")
    calcs = (input_data - np.mean(input_data, axis=0)) / (np.std(input_data, axis=0))
    comparison = ret == calcs
    assert comparison.all()


def test_scaler_function_scale(data_highly_correlated_dataframe):
    input_data = data_highly_correlated_dataframe["array"]
    ret = scaler(input_data=input_data, scale_type="scale")
    calcs = input_data - np.mean(input_data, axis=0)
    comparison = ret == calcs
    assert comparison.all()
