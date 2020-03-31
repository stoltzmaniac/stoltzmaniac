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
