import logging
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
        try:
            self.raw_data = np.array(input_data)
        except Exception as e:
            print(f"Input was not able to be converted to np.ndarray via np.array()")
            print(e)
        if not isinstance(self.raw_data, np.ndarray):
            raise TypeError(
                f"After calling np.array(input_data), \n"
                f"the result was not an nd.array and was of type {type(self.raw_data)}"
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


class TestTrainSplitData:
    def __init__(self, input_data: np.ndarray, train_split: float, seed: int):
        """
        Splits data into user specified sizes for test and train sets through randomization using np.seed
        Parameters
        ----------
        input_data: np.ndarray
        train_split: float for percentage of set to be used as training (represented as a decimal 0 < n < 1)
        seed: int to be used as a random seed for reproducibility
        """
        self.raw_data = input_data
        self.train_split = train_split
        if not isinstance(seed, int):
            raise TypeError(
                f"Seed must be an int. Currently the input is of type: {type(seed)}"
            )
        if not seed > 0:
            raise ValueError(f"Seed must be greater than 0, it is currently: {seed}")
        self.train_data = None
        self.test_data = None
        np.random.seed(seed=seed)
        self.split()

    def split(self):
        """
        Separates the data and splits it accordingly, utilizes the random seed
        Results in self.train_data and self.test_data
        Returns
        -------
        """
        if not isinstance(self.train_split, float):
            raise TypeError(
                f"train_split must be a float type and is currently: {type(self.train_split)}"
            )
        if not 0 < self.train_split < 1:
            raise ValueError(
                f"train_split must be between 0 and 1 (not including), it is {self.train_split}"
            )

        # Find indices for test / train split to use in selection from array
        split_row = round(self.raw_data.shape[0] * self.train_split)
        indices = np.random.permutation(self.raw_data.shape[0])
        train_idx, test_idx = indices[:split_row], indices[split_row:]

        # Ensure that both test and train data sets have more than one row of data
        if len(train_idx) < 1 or len(test_idx) < 1:
            raise ValueError(
                f"Train / Test break does not result in more than one row in test / train\n"
                f"len(train_idx) = {len(train_idx)}\n"
                f"len(test_idx) = {len(test_idx)}"
            )

        # Set select data and use ArrayData type to ensure adherence to standards
        self.train_data = ArrayData(self.raw_data[train_idx, :]).raw_data
        self.test_data = ArrayData(self.raw_data[test_idx, :]).raw_data


class ScaleData:
    def __init__(self, input_data: np.ndarray, scale_type: str):
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

        # Set statistics to be used in preprocessing - needed during preprocessing for predictions
        self.x_array_mean = np.mean(self.x_data, axis=0)
        self.x_array_max = np.max(self.x_data, axis=0)
        self.x_array_min = np.min(self.x_data, axis=0)

        # A standard deviation may not have a value due to square root, set to none in these cases
        try:
            self.x_array_std = np.std(self.raw_data, axis=0)
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
        if self.scale_type not in ["normalize", "standardize", "min_max", "scale"]:
            if self.scale_type:
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
        elif self.scale_type == "standardize":
            scaled_data = (x_data - self.x_array_mean) / self.x_array_std
        elif self.scale_type == "scale":
            scaled_data = x_data - self.x_array_mean
        else:
            scaled_data = x_data
        return scaled_data


class LinearRegression:
    def __init__(self, input_data, train_split=0.7, scale_type=None, seed=123):
        """
        Create and train model for linear regression, single or multivariate
        Parameters
        ----------
        input_data: can be of any type easily converted to np.ndarray
        train_split: float described in TestTrainSplitData model
        scale_type: str described in ScaleData model
        seed: flat described in TestTrainSplitData model
        """
        self.raw_data = input_data
        self.seed = seed
        self.train_split = train_split
        self.scale_type = scale_type

        # Set data to ArrayData type in order to ensure it passes requirements
        self.array_data = ArrayData(self.raw_data).raw_data

        # Split data for test / train
        self.split_data = TestTrainSplitData(
            input_data=self.array_data, train_split=self.train_split, seed=self.seed
        )

        # Set scaling parameters
        self.scaler = ScaleData(self.split_data.train_data, self.scale_type)

        # Scale TRAIN data only
        self.split_data.train_data = self.scaler.scale(self.scaler.x_data)

        # Fit linear regression model
        self.fit()

    def preprocess(self, data):
        """
        Preprocessing will clean and scale data that is to be used in
        Parameters
        ----------
        data

        Returns
        -------

        """
        array_data = ArrayData(data).raw_data
        scaled_data = self.scaler.scale(array_data)
        return ArrayData(scaled_data).raw_data

    def fit(self):
        x = self.split_data.train_data
        y = self.scaler.y_data
        # Add ones for intercept
        A = np.c_[x, np.ones(len(x))]
        self.betas = np.linalg.lstsq(A, y, rcond=None)[0]

    def predict(self, input_data):
        x_pre = self.preprocess(input_data)
        x = np.c_[x_pre, np.ones(len(x_pre))]
        return x.dot(self.betas)
