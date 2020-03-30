import numpy as np
import pandas as pd
from stoltzmaniac.data_handler.base import Base
from stoltzmaniac.utils.convert import check_expected_type, pd_dataframe_to_ndarray


class Converter(Base):
    def __init__(self, input_data):
        """
        Checks data types and converts input data to numpy array

        Parameters
        ----------
        input_data
        """
        super().__init__(input_data=input_data)
        self.data = np.ndarray

        # Check data type and convert to numpy array if required
        if check_expected_type(self.input_data, np.ndarray):
            self.data = input_data
        elif check_expected_type(self.input_data, pd.DataFrame):
            self.data = pd_dataframe_to_ndarray(input_data)
        else:
            raise TypeError("Input object must be pd.DataFrame or np.ndarray")
