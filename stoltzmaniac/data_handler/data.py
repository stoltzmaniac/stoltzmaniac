import numpy as np
from .utils import converter, describer, scaler, splitter, label_encoder, label_decoder


class Data:
    def __init__(
        self,
        data,
        target_column: int = None,
        train_split=0.7,
        scale_type=None,
        seed=123,
    ):
        """
        Data Class for data input, houses raw input
        Should be useful for any data that can be put into np.ndarray structure
        Parameters
        ----------
        data formats to be added, currently np.ndarray or pd.DataFrame
        """
        # Convert pd.DataFrame (currently only setup), will pass np.ndarray
        self.data = converter(data)
        self.data_train = np.array([])
        self.data_test = np.array([])
        self.train_split = train_split
        self.seed = seed
        self.scale_type = scale_type
        self.target_column = target_column  # defines response / target variable

    def label_encode(self):
        ret = label_encoder(self.data)
        self.data_encoded = ret["encoded_data"]
        self.data_labels = ret["encoded_labels"]

    def label_decode(self):
        self.data_decoded = label_decoder(self.data_encoded, self.data_labels)

    def split(self):
        """
        Create test / train data sets based off of train_split and seed
        """
        split_data = splitter(
            input_data=self.data, train_split=self.train_split, seed=self.seed
        )
        self.data_train = split_data["train"]
        self.data_test = split_data["test"]

    @staticmethod
    def scale(input_data: np.ndarray, scale_type: str = "normalize"):
        return scaler(input_data=input_data, scale_type=scale_type)

    # def _scale_train(self):
    #     """
    #     Required to ensure model retains training scale
    #     Returns
    #     -------
    #     """
    #     descriptive_stats = describer(self.data_train)
    #     self.data_train_max = descriptive_stats['max']
    #     self.data_train_mean = descriptive_stats['mean']
    #     self.data_train_min = descriptive_stats['min']
    #     self.data_train_std = descriptive_stats['std']
    #     self.data_train_data = scaler(self.data_train, self.scale_type)

    # def scale(input_data: np.ndarray, scale_type: str = "normalize"):
    #     self.sca
    #     return scaler(input_data=input_data, scale_type=scale_type)
