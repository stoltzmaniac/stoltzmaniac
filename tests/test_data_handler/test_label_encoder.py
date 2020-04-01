import pytest
import numpy as np
from stoltzmaniac.data_handler.utils import label_encoder


def test_label_encoder_function_basics():
    # TODO: function does not handle for mixed data types
    with pytest.raises(TypeError):
        label_encoder()


def test_label_encoder_function_consistent_data_types_mixed():
    input_data = np.array(
        [
            ["hi", "there", 4.3],
            ["scott", "james", 3.1],
            ["hi", "scott", 0.1],
            ["please", "there", 10.2],
        ]
    )
    ret = label_encoder(input_data)

    comp1 = ret["encoded_data"] == np.array(
        [[0.0, 2.0, 4.3], [2.0, 0.0, 3.1], [0.0, 1.0, 0.1], [1.0, 2.0, 10.2]]
    )
    assert comp1.all()
    assert ret["encoded_labels"] == [
        {0: "hi", 2: "scott", 1: "please"},
        {2: "there", 0: "james", 1: "scott"},
        {},
    ]


def test_label_encoder_function_consistent_data_types_string():
    # Single
    input_data = np.array(["hi", "there", "scott", "hi"])
    ret = label_encoder(input_data)

    comp1 = ret["encoded_data"] == np.array([[0], [2], [1], [0]])
    assert comp1.all()
    assert ret["encoded_labels"] == [{0: "hi", 2: "there", 1: "scott"}]

    # Multiple
    input_data = np.array(
        [["hi", "there", "scott", "hi"], ["please", "there", "yes", "no"]]
    )
    ret = label_encoder(input_data)
    comp1 = ret["encoded_data"] == np.array([[0, 0, 0], [1, 0, 1]])
    assert comp1.all()
    assert ret["encoded_labels"] == [
        {0: "hi", 1: "please"},
        {0: "there"},
        {0: "scott", 1: "yes"},
    ]


def test_label_encoder_function_consistent_data_types_numeric():
    # Integer
    input_data = np.array([1, 10, 3])
    ret = label_encoder(input_data)
    comp1 = ret["encoded_data"] == np.array([[1], [10], [3]])
    assert comp1.all()
    assert ret["encoded_labels"] == [{}]

    # Float
    input_data = np.array([1.1, 10.4, 3.4])
    ret = label_encoder(input_data)
    comp1 = ret["encoded_data"] == np.array([[1.1], [10.4], [3.4]])
    assert comp1.all()
    assert ret["encoded_labels"] == [{}]
