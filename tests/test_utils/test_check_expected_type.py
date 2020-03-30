import pytest
import pandas as pd
from stoltzmaniac.utils.check_types import check_expected_type


def test_check_expected_type():
    """
    Instantiates both a pd.DataFrame and a list to check, looks for exception as well
    :return:
    """
    # Newly defined DataFrame should match
    data = pd.DataFrame({"foo": ["bar", "is", "silly"]})
    assert check_expected_type(data, pd.DataFrame) is True

    # lists should match
    data = ["1", "2"]
    assert check_expected_type(data, list) is True

    # Unlike types should not match, should throw TypeError
    data = ["1", "2"]
    assert check_expected_type(data, pd.DataFrame) is False
