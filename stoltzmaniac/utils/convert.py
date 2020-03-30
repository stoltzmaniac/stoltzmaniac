import pandas as pd
import numpy as np
from ..utils.check_types import check_expected_type


def pd_dataframe_to_ndarray(df: pd.DataFrame):
    """
    Convert a pandas dataframe to a numpy array
    Parameters
    ----------
    df
        pd.DataFrame

    Returns
    -------
    np.ndarray

    """

    check_expected_type(df, pd.DataFrame)
    return np.array(df)
