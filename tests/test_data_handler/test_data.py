import pytest
import numpy as np
import pandas as pd
from stoltzmaniac.data_handler.data import Data


def test_data_class_basics(data_highly_correlated_dataframe):
    """
    Nothing to test yet
    :return:
    """
    with pytest.raises(TypeError):
        Data()

    data_class_from_array = Data(data=data_highly_correlated_dataframe["array"])
    assert type(data_class_from_array.data) == np.ndarray
    assert type(data_class_from_array.data_test) == np.ndarray
    assert type(data_class_from_array.data_train) == np.ndarray
    assert data_class_from_array.train_split == 0.7
    assert data_class_from_array.seed == 123


def test_data_from_array_methods_split(data_highly_correlated_dataframe):
    data_class_from_array = Data(data=data_highly_correlated_dataframe["array"])
    data_class_from_array.split()
    assert data_class_from_array.data_train.shape == (107, 3)
    assert data_class_from_array.data_test.shape == (46, 3)


def test_data_from_array_methods_scale(data_highly_correlated_dataframe):
    data_class_from_array = Data(data=data_highly_correlated_dataframe["array"])
    ret = data_class_from_array.scale(
        input_data=data_class_from_array.data, scale_type="scale"
    )
    calcs = data_class_from_array.data - np.mean(data_class_from_array.data, axis=0)
    comparison = ret == calcs
    assert comparison.all()


def test_data_from_array_methods_label_encode():
    input_data = np.array(
        [
            ["oh", "hi", 4.3, "there"],
            ["me", "scott", 3.1, "james"],
            ["oh", "hi", 0.1, "scott"],
            ["my", "please", 10.2, "there"],
        ]
    )
    d = Data(input_data)
    d.label_encode()

    comp1 = d.data_encoded == np.array(
        [[2, 0, 4.3, 2], [0, 2, 3.1, 0], [2, 0, 0.1, 1], [1, 1, 10.2, 2]]
    )
    assert comp1.all()
    assert d.data_labels == [
        {2: "oh", 0: "me", 1: "my"},
        {0: "hi", 2: "scott", 1: "please"},
        {},
        {2: "there", 0: "james", 1: "scott"},
    ]


def test_data_from_array_methods_label_decode():
    input_data = np.array(
        [
            ["oh", "hi", 4.3, "there"],
            ["me", "scott", 3.1, "james"],
            ["oh", "hi", 0.1, "scott"],
            ["my", "please", 10.2, "there"],
        ]
    )
    d = Data(input_data)
    d.label_encode()
    d.label_decode()

    comp1 = input_data == d.data_decoded
    assert comp1.all()
