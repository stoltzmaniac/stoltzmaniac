import numpy as np


# TODO: fix scaling types, calculations are not right


class ScaleData:
    def __init__(self, data: np.ndarray, scale_type: str):
        """
        Scales data, may use ONE scale_type: 'normalize', 'standardize', 'min_max', 'scale'
        Parameters
        ----------
        data: should be train data only
        scale_type: type of scaling desired for training data
        """
        self.original_data = data  # used to store raw values for user reference
        self.scale_type = scale_type

        if not isinstance(self.scale_type, str) and self.scale_type is not None:
            raise TypeError(
                f"scale_type must be None or of type str, it is: {type(scale_type)}"
            )

        # Set statistics to be used in preprocessing - needed during preprocessing for predictions
        self.array_mean = np.mean(self.original_data, axis=0)
        self.array_max = np.max(self.original_data, axis=0)
        self.array_min = np.min(self.original_data, axis=0)
        self.array_std = np.std(self.original_data, axis=0)

        self.original_scaled_data = self.scale(data=self.original_data)

    def scale(self, data: np.ndarray):
        """
        Scales data to be used in both training and prediction. Scale should handle for a number of cases
        This function must be run on ALL data being passed, even if scaling will be of None type
        Parameters
        ----------
        data: np.ndarray of ONLY predictor values

        Returns
        -------
        np.ndarray of scaled data or returns data if self.scale_type is None
        """
        # Ensure that scale type is handled by function
        if (
            self.scale_type not in ["normalize", "standardize", "min_max", "scale"]
            and self.scale_type
        ):
            raise ValueError(
                f"scale_type {self.scale_type} not in ['normalize', 'standardize', 'min_max', 'scale']"
            )
        if data.shape == (1, 0):
            raise ValueError("data shape is None")

        # Start scaling process based off of self.scale_type
        if self.scale_type == "min_max":
            scaled_data = (data - self.array_min) / (self.array_max - self.array_mean)
        elif self.scale_type == "normalize":
            scaled_data = (data - self.array_mean) / (self.array_max - self.array_min)
        elif self.scale_type == "scale":
            scaled_data = data - self.array_mean
        elif self.scale_type == "standardize":
            scaled_data = (data - self.array_mean) / self.array_std
        else:
            scaled_data = data
        return scaled_data
