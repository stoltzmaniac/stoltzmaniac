import numpy as np

from stoltzmaniac.data_handler.base import ArrayData
from stoltzmaniac.data_handler.clean_data import CleanData
from stoltzmaniac.data_handler.scale_data import ScaleData
from stoltzmaniac.data_handler.test_train_split_data import TrainTestSplitData


class LinearRegression:
    def __init__(self, input_data, train_split=0.7, scale_type=None, seed=123):
        """
        Create and train model for linear regression, single or multivariate
        Parameters
        ----------
        input_data: can be of any type easily converted to np.ndarray
        train_split: float described in TrainTestSplitData model
        scale_type: str described in ScaleData model
        seed: flat described in TrainTestSplitData model
        """
        self.raw_data = input_data
        self.seed = seed
        self.train_split = train_split
        self.scale_type = scale_type

        # Set data to ArrayData type in order to ensure it passes requirements
        self.array_data = ArrayData(self.raw_data).raw_data
        self.clean_data = CleanData(self.array_data).clean_data

        # Split data for test / train
        self.split_data = TrainTestSplitData(
            input_data=self.clean_data, train_split=self.train_split, seed=self.seed
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
        data: np.ndarray representing predictor variables ("x_data")

        Returns
        -------
        np.ndarray
        """
        array_data = ArrayData(data).raw_data
        scaled_data = self.scaler.scale(array_data)
        return ArrayData(scaled_data).raw_data

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
