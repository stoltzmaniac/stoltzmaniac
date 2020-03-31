import numpy as np
from .utils import converter, scaler, splitter


class Data:
    def __init__(self, data, train_split=0.7, seed=123):
        """
        Data Class for data input, houses raw input
        Should be useful for any data that can be put into np.ndarray structure
        Parameters
        ----------
        data formats to be added, currently np.ndarray or pd.DataFrame
        """
        # Convert pd.DataFrame (currently only setup), will pass np.ndarray
        self.data = converter(data)
        self.data_train = np.ndarray
        self.data_test = np.ndarray
        self.train_split = train_split
        self.seed = seed

    def split(self):
        split_data = splitter(
            input_data=self.data, train_split=self.train_split, seed=self.seed
        )
        self.data_train = split_data["train"]
        self.data_test = split_data["test"]

    @staticmethod
    def scale(input_data: np.ndarray, scale_type: str = "normalize"):
        return scaler(input_data=input_data, scale_type=scale_type)
