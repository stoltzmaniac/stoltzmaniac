import numpy as np


class ArrayData:
    def __init__(self, input_data):
        """
        BaseData Class for data input, houses raw input
        Should be useful for any data that can be put into np.ndarray structure
        Parameters
        ----------
        input_data: if supervised learning, last column must be the target/response variable
        """
        # TODO: create way to generalize input_data to accept more than continuous variable and formats
        try:
            self.raw_data = np.array(input_data, dtype=np.float)
        except Exception as e:
            raise ValueError(
                f"Input was not able to be converted to np.ndarray via np.array(input_data, dtype=np.float)"
            )

        # Convert any 1 dimensional array to a column format
        if self.raw_data.ndim == 1:
            self.raw_data = self.raw_data[np.newaxis]

    def __str__(self):
        ret_str = (
            f"Data specs:\n"
            f"type: {type(self.raw_data)}\n"
            f"shape: {self.raw_data.shape}\n"
            f"ndim: {self.raw_data.ndim}"
        )
        return ret_str
