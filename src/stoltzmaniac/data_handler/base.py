import numpy as np


# TODO: add some kind of encoding for categorical data


class BaseData:
    def __init__(self, input_data):
        """
        BaseData Class for data input, houses raw input
        Must be numpy.ndarray
        Parameters
        ----------
        input_data: if supervised learning, last column must be the target/response variable
        """

        if not isinstance(input_data, np.ndarray):
            raise ValueError(
                f"input_data is not of type numpy.ndarray and is of type {type(input_data)}"
            )

        self.data = input_data

        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

        if self.data.dtype not in [np.dtype("int64"), np.dtype("float64")]:
            raise ValueError(
                f"input_data needs to be either np.float64 or np.int64, input_data is currently: {self.data.dtype}"
            )

    def __str__(self):
        ret_str = (
            f"Data specs:\n"
            f"type: {type(self.data)}\n"
            f"shape: {self.data.shape}\n"
            f"ndim: {self.data.ndim}"
        )
        return ret_str


class UnsupervisedData:
    def __init__(self, X: np.ndarray):
        """
        Base class for unsupervised data, should not have a target variable.
        Parameters
        ----------
        X is an array of dimensions
        """
        self.X = BaseData(X).data


class SupervisedData:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Base class for supervised data, requires a target.
        Parameters
        ----------
        X is an array of dimensions
        y is a target array
        """
        self.X = BaseData(X).data
        self.y = BaseData(y).data

        if self.y.shape[1] != 1:
            raise ValueError(
                f"Target variable (y) needs to only have 1 column, currently y has {self.y.shape[1]}"
            )

        if self.X.shape[0] != self.y.shape[0]:
            raise AttributeError(
                f"Variables X and y need to have the same number of rows. currently X has {self.X.shape[0]} and y has {self.y.shape[0]} rows"
            )


class RegressionData:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Input is based off of SupervisedData class, checks basics
        Parameters
        ----------
        X is an array of dimensions
        y is a target array of floats
        """
        supervised_data = SupervisedData(X=X, y=y)
        self.X = supervised_data.X
        self.y = supervised_data.y

        # Check to ensure `y` is for regression and not classification
        if self.y.dtype != np.dtype("float64"):
            raise ValueError(
                f"Target variable (y) must be of dtype float64, currently y is of dtype {self.y.dtype}"
            )


class ClassificationData:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Input is based off of SupervisedData class, checks basics
        Parameters
        ----------
        X is an array of dimensions
        y is a target array of integers
        """
        supervised_data = SupervisedData(X=X, y=y)
        self.X = supervised_data.X
        self.y = supervised_data.y

        if self.y.shape[1] != 1:
            raise ValueError(
                f"Target variable (y) needs to only have 1 column, currently y has {self.y.shape[1]}"
            )

        # Check to ensure `y` is for regression and not classification
        if self.y.dtype != np.dtype("int64"):
            raise ValueError(
                f"Target variable (y) must be of dtype int64, currently y is of dtype {self.y.dtype}"
            )
