from typing import Union
import logging

import numpy as np


class ScaleData:
    def __init__(self, input_data: np.ndarray, scale_type: Union[str, None]):
        """
        Scales data, may use ONE scale_type: 'normalize', 'standardize', 'min_max', 'scale'
        Parameters
        ----------
        input_data
        scale_type
        """
        self.raw_data = input_data  # used to store raw values for user reference
        self.scale_type = scale_type
        self.x_data = self.raw_data[:, :-1]
        self.y_data = self.raw_data[:, -1]

        if not isinstance(self.scale_type, str) and self.scale_type is not None:
            raise TypeError(
                f"scale_type must be None or of type str, it is: {type(scale_type)}"
            )

        # Set statistics to be used in preprocessing - needed during preprocessing for predictions
        self.x_array_mean = np.mean(self.x_data, axis=0)
        self.x_array_max = np.max(self.x_data, axis=0)
        self.x_array_min = np.min(self.x_data, axis=0)

        # A standard deviation may not have a value due to square root, set to none in these cases
        try:
            self.x_array_std = np.std(self.x_data, axis=0)
        except TypeError as e:
            logging.warning("array_std not possible due to sqrt")
            self.x_array_std = None

        if self.scale_type == "standardized" and self.x_array_std is None:
            raise ValueError(
                f"Cannot use scale_type 'standardize' because standard deviation is of None type"
            )

    def scale(self, x_data: np.ndarray):
        """
        Scales data to be used in both training and prediction. Scale should handle for a number of cases
        This function must be run on ALL data being passed, even if scaling will be of None type
        Parameters
        ----------
        x_data: np.ndarray of ONLY predictor values

        Returns
        -------
        np.ndarray of scaled data or returns x_data if self.scale_type is None
        """
        # Ensure that scale type is handled by function
        if (
            self.scale_type not in ["normalize", "standardize", "min_max", "scale"]
            and self.scale_type
        ):
            raise ValueError(
                f"scale_type {self.scale_type} not in ['normalize', 'standardize', 'min_max', 'scale']"
            )
        if x_data.shape is None:
            raise ValueError("x_data shape is None")

        # Star scaling process based off of self.scale_type
        if self.scale_type == "min_max":
            scaled_data = (x_data - self.x_array_min) / (
                self.x_array_max - self.x_array_mean
            )
        elif self.scale_type == "normalize":
            scaled_data = (x_data - self.x_array_mean) / (
                self.x_array_max - self.x_array_min
            )
        elif self.scale_type == "scale":
            scaled_data = x_data - self.x_array_mean
        elif self.scale_type == "standardize":
            scaled_data = (x_data - self.x_array_mean) / self.x_array_std
        else:
            scaled_data = x_data
        return scaled_data
