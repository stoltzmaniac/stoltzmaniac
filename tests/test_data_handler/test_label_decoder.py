import pytest
import numpy as np
from stoltzmaniac.data_handler.utils import label_encoder, label_decoder


def test_label_decoder_function_basics():
    # TODO: function does not handle for mixed data types
    with pytest.raises(TypeError):
        label_decoder()


def test_label_decoder_function_consistent_data_types_mixed():
    input_data = np.array(
        [
            ["hi", "there", 4.3],
            ["scott", "james", 3.1],
            ["hi", "scott", 0.1],
            ["please", "there", 10.2],
        ]
    )
    ret = label_encoder(input_data)
    dec = label_decoder(ret["encoded_data"], ret["encoded_labels"])
    comp1 = dec == input_data
    assert comp1.all()


# def test_label_decoder_function_consistent_data_types_string():
#     # Single
#     input_data = np.array(["hi", "there", "scott", "hi"])
#     ret = label_encoder(input_data)
#
#     dec = label_decoder(ret["encoded_data"], ret["encoded_labels"])
#     comp1 = dec == input_data
#     print(input_data)
#     print(type(input_data))
#     print(dec)
#     print(type(dec))
#     assert comp1.all()
#
#     # Multiple
#     input_data = np.array(
#         [["hi", "there", "scott", "hi"], ["please", "there", "yes", "no"]]
#     )
#     ret = label_encoder(input_data)
#     dec = label_decoder(ret["encoded_data"], ret["encoded_labels"])
#     comp1 = dec == input_data
#     assert comp1.all()
#
#
# def test_label_decoder_function_consistent_data_types_numeric():
#     # Integer
#     input_data = np.array([1, 10, 3])
#     ret = label_encoder(input_data)
#     dec = label_decoder(ret["encoded_data"], ret["encoded_labels"])
#     comp1 = dec == input_data
#     assert comp1.all()
#
#     # Float
#     input_data = np.array([1.1, 10.4, 3.4])
#     ret = label_encoder(input_data)
#     dec = label_decoder(ret["encoded_data"], ret["encoded_labels"])
#     comp1 = dec == input_data
#     assert comp1.all()
