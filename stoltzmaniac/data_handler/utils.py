import numpy as np
import pandas as pd

from stoltzmaniac.utils.check_types import check_expected_type
from stoltzmaniac.utils.convert import pd_dataframe_to_ndarray


def converter(input_data) -> np.ndarray:
    """
    Converts data to numpy array
    Parameters
    ----------
    input_data: np.ndarray, pd.DataFrame

    Returns
    -------
    np.ndarray

    """
    # Check data type and convert to numpy array if required
    if check_expected_type(input_data, np.ndarray):
        return input_data
    elif check_expected_type(input_data, pd.DataFrame):
        return pd_dataframe_to_ndarray(input_data)
    else:
        raise TypeError("Input object must be pd.DataFrame or np.ndarray")


def scaler(input_data: np.ndarray, scale_type: str = None) -> np.ndarray:
    """
    Data preprocessing step to scale
    Parameters
    ----------
    input_data np.ndarray
    scale_type str in ["normalize", "standardize", "min_max", "scale"]

    Returns
    -------

    """
    if scale_type not in ["normalize", "standardize", "min_max", "scale"]:
        if scale_type:
            raise ValueError(
                f"scale_type {scale_type} not in ['normalize', 'standardize', 'min_max', 'scale']"
            )
    if not input_data.shape:
        raise ValueError("input_data shape is None")
    array_mean = np.mean(input_data, axis=0)
    array_std = np.std(input_data, axis=0)
    array_max = np.max(input_data, axis=0)
    array_min = np.min(input_data, axis=0)

    if scale_type == "min_max":
        scaled_data = (input_data - array_min) / (array_max - array_mean)
    elif scale_type == "normalize":
        scaled_data = (input_data - array_mean) / (array_max - array_min)
    elif scale_type == "standardize":
        scaled_data = (input_data - array_mean) / array_std
    elif scale_type == "scale":
        scaled_data = input_data - array_mean
    else:
        scaled_data = input_data
    return scaled_data


def splitter(input_data: np.ndarray, train_split: float = 0.7, seed: int = 123) -> dict:
    """
    Separates the data and splits it accordingly, utilizes the random seed
    Parameters
    ----------
    input_data np.ndarray
    train_split float
    seed int

    Returns
    -------
    {'test': np.ndarray, 'train': np.ndarray}
    """

    if not check_expected_type(input_data, np.ndarray):
        raise TypeError
    if type(seed) != int:
        raise ValueError(f"seed value not an int, it is {seed}")
    if not type(train_split) == float:
        raise ValueError(f"train_split not a float, it is {train_split}")
    if not 0 < train_split < 1:
        raise ValueError(
            f"train_split must be between 0 and 1 (not including), it is {train_split}"
        )

    np.random.seed(seed=seed)
    indices = np.random.permutation(input_data.shape[0])
    split_row = round(input_data.shape[0] * train_split)
    train_idx, test_idx = indices[:split_row], indices[split_row:]
    train_data, test_data = input_data[train_idx, :], input_data[test_idx, :]
    return {"test": test_data, "train": train_data}
