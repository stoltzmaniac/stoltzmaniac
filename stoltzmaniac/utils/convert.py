import pandas as pd
import numpy as np
from .check_types import check_expected_type


def pd_dataframe_to_ndarray(df: pd.DataFrame):
    """
    Convert a pandas dataframe to a numpy array
    Parameters
    ----------
    df
        pd.DataFrame

    Returns
    -------
    Numpy representation of pandas DataFrame
    np.ndarray

    """

    if check_expected_type(df, pd.DataFrame):
        return np.array(df)
    else:
        raise TypeError(
            f"Expected type does not match pd.DataFrame and is of type: {type(df)}"
        )


def convert_column_to_numeric(data: np.ndarray):
    """
    Check types of data and convert (in case numeric data is stored as strings)
    Parameters
    ----------
    data

    Returns
    -------
    data, is_numeric
    """
    data = data.astype(str)  # in order to avoid float being rounded as int
    try:
        data = data.astype(int)
        return data, True
    except Exception as e:
        # print(e)
        pass
    try:
        data = data.astype(float)
        return data, True
    except Exception as e:
        # print(e)
        pass
    return np.unique(data, return_inverse=True)[1], False
