import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def data_highly_correlated_dataframe() -> dict:
    """
    Setup test data for
    :return:
    """
    df = pd.read_csv("tests/data/highly_correlated.csv")
    yield {
        "dataframe": df,
        "array": np.array(df),
    }
    return print("data_highly_correlated_dataframe fixture finished.")


@pytest.fixture
def data_loosely_correlated_dataframe() -> dict:
    """
    Setup test data for
    :return:
    """
    df = pd.read_csv("tests/data/loosely_correlated.csv")
    yield {
        "dataframe": df,
        "array": np.array(df),
    }
    return print("data_loosely_correlated_dataframe fixture finished.")
