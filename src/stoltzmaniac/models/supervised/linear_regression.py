import numpy as np

from stoltzmaniac.data_handler.base import BaseData, RegressionData
from stoltzmaniac.data_handler.scale_data import ScaleData
from stoltzmaniac.data_handler.test_train_split_data import TrainTestSplitData


class LinearRegression:
    def __init__(
        self, X: np.ndarray, y: np.ndarray, train_split=0.7, scale_type=None, seed=123
    ):
        """
        Create and train model for linear regression, single or multivariate
        Parameters
        ----------
        X: array of predictor variables
        y: array of target variable
        train_split: float described in TrainTestSplitData model
        scale_type: str described in ScaleData model
        seed: flat described in TrainTestSplitData model
        """

        self.scale_type = scale_type
        self.data = RegressionData(X, y)
        self.X = self.data.X
        self.y = self.data.y

        X_split = TrainTestSplitData(X, train_split=train_split, seed=seed)
        y_split = TrainTestSplitData(y, train_split=train_split, seed=seed)
        self.X_train = X_split.train_data
        self.X_test = X_split.test_data
        self.y_train = y_split.train_data
        self.y_test = y_split.test_data

        # Set scaling parameters, assigns fixed scaling parameters
        self.scaler = ScaleData(self.X_train, scale_type=self.scale_type)

        # Fit linear regression model
        self.fit()

    def preprocess(self, data: np.ndarray):
        """
        Preprocessing will clean and scale data that is to be used in
        Parameters
        ----------
        data: np.ndarray representing predictor variables ("x_data")

        Returns
        -------
        np.ndarray
        """
        array_data = BaseData(data).data
        scaled_data = self.scaler.scale(array_data)
        return BaseData(scaled_data).data

    def fit(self):
        """
        Fits the model with self.train_data and creates self.betas to represent coefficients
        Last self.beta represents the intercept coefficent
        """
        X = self.preprocess(self.X_train)
        y = self.y_train
        # Add ones for intercept
        A = np.c_[X, np.ones(len(X))]
        self.betas = np.linalg.lstsq(A, y, rcond=None)[0]

    def predict(self, data: np.ndarray):
        """
        Predict results based off of predictor variables, must match data format from initial training data
        Parameters
        ----------
        data: array for data to be predicted, will automatically scale

        Returns
        -------
        np.ndarray of response variable
        """
        new_data = BaseData(data).data
        if new_data.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"predict data must be the same number of columns as the original X data. # of Columns: Original X = {self.X.shape[1]}, current data = {new_data.shape[1]}"
            )

        x_pre = self.preprocess(data)
        x = np.c_[x_pre, np.ones(len(x_pre))]
        return x.dot(self.betas)
