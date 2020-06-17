import numpy as np

from stoltzmaniac.data_handler.base import BaseData


class CleanData:
    def __init__(self, input_data: np.ndarray):
        """
        Checks column types and provides NA / NaN / Inf detection
        Encode data by converting categorical to integers, keep encoding for decoding
        Parameters
        ----------
        input_data: np.ndarray to clean and encode
        """
        self.raw_data = input_data
        # self.na_array = np.isnan(self.data)
        # self.inf_array = np.isinf(self.data)
        self.clean()

    def clean(self):
        # TODO: Create a cleaning function
        self.clean_data = BaseData(self.raw_data).data
