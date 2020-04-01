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
        ret = input_data
    elif check_expected_type(input_data, pd.DataFrame):
        ret = pd_dataframe_to_ndarray(input_data)
    else:
        raise TypeError("Input object must be pd.DataFrame or np.ndarray")

    # Array must have shape (x, 1) and not (x,) -- necessary for indexing and operations
    if input_data.ndim == 1:
        ret = np.array([[i] for i in input_data])
    return ret


def describer(input_data: np.ndarray) -> dict:
    """
    Calculates descriptive statistics
    Parameters
    ----------
    input_data np.ndarray

    Returns
    -------
    """
    array_mean = np.mean(input_data, axis=0)
    array_std = np.std(input_data, axis=0)
    array_max = np.max(input_data, axis=0)
    array_min = np.min(input_data, axis=0)
    return {"max": array_max, "mean": array_mean, "min": array_min, "std": array_std}


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

    descriptive_stats = describer(input_data)
    array_max = descriptive_stats["max"]
    array_mean = descriptive_stats["mean"]
    array_min = descriptive_stats["min"]
    array_std = descriptive_stats["std"]

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


def label_encoder(input_data: np.ndarray) -> dict:
    """
    Encode strings as integers within array and keep track of their original values in a dict
    Parameters
    ----------
    input_data np.ndarray

    Returns
    -------

    """

    def convert_numeric(data: np.ndarray):
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

    my_data_list = []
    my_numeric_list = []

    for i in range(input_data.shape[1]):
        d = input_data[:, i]
        ret_d, is_numeric = convert_numeric(d)
        my_data_list.append(ret_d)
        my_numeric_list.append(is_numeric)
        encoded_data = np.column_stack(my_data_list)

    if "encoded_data" in locals():
        pass
    else:
        raise ValueError(
            "encoded_data does not exist in convert_numeric function of label_encoder"
        )

    encoded_labels = []
    i = 0
    for orignal_column, encoded_column in zip(
        np.column_stack(input_data), np.column_stack(encoded_data)
    ):
        d = {}
        if not my_numeric_list[i]:
            for real_element, encoded_element in zip(orignal_column, encoded_column):
                if not np.char.isnumeric(real_element):
                    d[int(encoded_element)] = real_element
        encoded_labels.append(d)
        i += 1

    return {"encoded_data": encoded_data, "encoded_labels": encoded_labels}


def label_decoder(data: np.ndarray, lookup: list):
    """
    Maps encoded labels back
    Parameters
    ----------
    data
    lookup

    Returns
    -------
    """
    decoder_list = []
    for i in range(data.shape[1]):
        if lookup[i]:
            d = np.vectorize(lookup[i].get)(data[:, i])
        else:
            d = data[:, i]
        decoder_list.append(d)
    ret = np.column_stack(decoder_list)
    return ret
