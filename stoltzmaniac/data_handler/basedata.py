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
    def __init__(self, input_data, train_split: float, seed: int):
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
        split_row = round(self.raw_data.shape[0] * self.train_split)
        indices = np.random.permutation(self.raw_data.shape[0])
        train_idx, test_idx = indices[:split_row], indices[split_row:]

        if len(train_idx) < 1 or len(test_idx) < 1:
            raise ValueError(
                f"Train / Test break does not result in more than one row in test / train\n"
                f"len(train_idx) = {len(train_idx)}\n"
                f"len(test_idx) = {len(test_idx)}"
            )
        self.train_data = ArrayData(self.raw_data[train_idx, :]).raw_data
        self.test_data = ArrayData(self.raw_data[test_idx, :]).raw_data


class ScaleData:
    def __init__(self, input_data: np.ndarray, scale_type: str = "standardize"):
        self.raw_data = input_data
        self.scale_type = scale_type
        self.x_data = self.raw_data[:, :-1]
        self.y_data = self.raw_data[:, -1]
        self.x_array_mean = np.mean(self.x_data, axis=0)
        self.x_array_max = np.max(self.x_data, axis=0)
        self.x_array_min = np.min(self.x_data, axis=0)

        try:
            self.x_array_std = np.std(self.raw_data, axis=0)
        except TypeError as e:
            logging.warning("array_std not possible due to sqrt")
            self.x_array_std = None

    def scale(self, x_data):
        # TODO: handle for 0 in data (divide by 0)
        if self.scale_type not in ["normalize", "standardize", "min_max", "scale"]:
            if self.scale_type:
                raise ValueError(
                    f"scale_type {self.scale_type} not in ['normalize', 'standardize', 'min_max', 'scale']"
                )
        if self.raw_data.shape is None:
            raise ValueError("input_data shape is None")

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
        self.raw_input_data = input_data
        self.seed = seed
        self.train_split = train_split
        self.scale_type = scale_type

        self.array_data = ArrayData(self.raw_input_data).raw_data
        self.split_data = TestTrainSplitData(
            input_data=self.array_data, train_split=self.train_split, seed=self.seed
        )
        self.scaler = ScaleData(self.split_data.train_data, self.scale_type)
        self.split_data.train_data = self.scaler.scale(self.scaler.x_data)
        self.fit()

    def preprocess(self, data):
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
