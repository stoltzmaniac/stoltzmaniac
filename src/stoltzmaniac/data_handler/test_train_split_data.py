import numpy as np

from stoltzmaniac.data_handler.base import BaseData


class TrainTestSplitData:
    def __init__(self, data: np.ndarray, train_split: float = 0.7, seed: int = 123):
        """
        Splits data into user specified sizes for test and train sets through randomization using np.seed
        Parameters
        ----------
        data: np.ndarray
        train_split: float for percentage of set to be used as training (represented as a decimal 0 < n < 1)
        seed: int to be used as a random seed for reproducibility
        """
        self.data = data
        self.train_split = train_split
        self.seed = seed

        if type(self.seed) != int:
            raise TypeError(
                f"Seed must be an int. Currently the input is of type: {type(self.seed)}"
            )
        if not self.seed > 0:
            raise ValueError(
                f"Seed must be greater than 0, it is currently: {self.seed}"
            )

        if not isinstance(self.train_split, float):
            raise TypeError(
                f"train_split must be a float type and is currently: {type(self.train_split)}"
            )
        if not 0 < self.train_split < 1:
            raise ValueError(
                f"train_split must be between 0 and 1 (not including), it is {self.train_split}"
            )

        # Finish setup
        self.train_data = None
        self.test_data = None
        np.random.seed(seed=self.seed)

        # Split the data upon instantiation
        self.split()

    def split(self):
        """
        Separates the data and splits it accordingly, utilizes the random seed
        Results in self.train_data and self.test_data
        Returns
        -------
        """
        # Find indices for test / train split to use in selection from array
        split_row = round(self.data.shape[0] * self.train_split)
        indices = np.random.permutation(self.data.shape[0])
        train_idx, test_idx = indices[:split_row], indices[split_row:]

        # Ensure that both test and train data sets have more than one row of data
        if len(train_idx) < 1 or len(test_idx) < 1:
            raise ValueError(
                f"Train / Test break does not result in more than one row in test / train\n"
                f"len(train_idx) = {len(train_idx)}\n"
                f"len(test_idx) = {len(test_idx)}"
            )

        # Set select data and use BaseData type to ensure adherence to standards
        self.train_data = BaseData(self.data[train_idx]).data
        self.test_data = BaseData(self.data[test_idx]).data
