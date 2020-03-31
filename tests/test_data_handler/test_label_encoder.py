import pytest
import numpy as np
from stoltzmaniac.data_handler.utils import label_encoder


def test_label_encoder_function_basics():
    # TODO: function does not handle for mixed data types
    with pytest.raises(TypeError):
        label_encoder()

    arr = np.array(
        [["hi", "there"], ["scott", "james"], ["hi", "scott"], ["please", "there"]]
    )
    ret = label_encoder(arr)
    comp1 = ret["encoded_data"] == np.array([[0, 2], [2, 0], [0, 1], [1, 2]])
    assert comp1.all()
    assert ret["encoded_labels"] == [
        {0: "hi", 2: "scott", 1: "please"},
        {2: "there", 0: "james", 1: "scott"},
    ]

    encoded_data = ret["encoded_data"]
    encoded_labels = ret["encoded_labels"]
    assert type(ret) == dict
    assert type(encoded_data) == np.ndarray
    assert type(encoded_labels) == list
