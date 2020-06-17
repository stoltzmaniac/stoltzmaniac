import numpy as np

from stoltzmaniac.data_handler.base import BaseData, RegressionData
from stoltzmaniac.data_handler.scale_data import ScaleData
from stoltzmaniac.data_handler.test_train_split_data import TrainTestSplitData


class LinearRegression:
    def __init__(self, X: np.ndarray, y: np.ndarray, train_split=0.7, scale_type=None, seed=123):
        """
        Create and train model for linear regression, single or multivariate
        Parameters
        ----------
        input_data: can be of any type easily converted to np.ndarray
        train_split: float described in TrainTestSplitData model
        scale_type: str described in ScaleData model
        seed: flat described in TrainTestSplitData model
        """

        self.scale_type = scale_type
        self.raw_data = RegressionData(X, y)
        self.X = self.raw_data.X
        self.Y = self.raw_data.y

        X_split = TrainTestSplitData(X, train_split=train_split, seed=seed)
        y_split = TrainTestSplitData(y, train_split=train_split, seed=seed)
        self.X_train = X_split.train_data
        self.X_test = X_split.test_data
        self.y_train = y_split.train_data
        self.y_test = y_split.test_data

        # Set scaling parameters
        self.scaler = ScaleData(self.X_train, scale_type=self.scale_type)

        # Fit linear regression model
        self.fit()

    def preprocess(self, data):
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
        x = self.split_data.train_data
        y = self.scaler.y_data
        # Add ones for intercept
        A = np.c_[x, np.ones(len(x))]
        self.betas = np.linalg.lstsq(A, y, rcond=None)[0]

    def predict(self, input_data):
        """
        Predict results based off of predictor variables, must match data format from initial training data
        Parameters
        ----------
        input_data

        Returns
        -------
        np.ndarray of response variable
        """
        x_pre = self.preprocess(input_data)
        x = np.c_[x_pre, np.ones(len(x_pre))]
        return x.dot(self.betas)
