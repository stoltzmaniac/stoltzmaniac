import numpy as np

from stoltzmaniac.data_handler.base import ArrayData


class TrainTestSplitData:
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
        self.seed = seed

        if not isinstance(self.seed, int):
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

        self.train_data = None
        self.test_data = None
        np.random.seed(seed=self.seed)
        self.split()

    def split(self):
        """
        Separates the data and splits it accordingly, utilizes the random seed
        Results in self.train_data and self.test_data
        Returns
        -------
        """
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
