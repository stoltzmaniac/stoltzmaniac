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
        self.seed = seed
        self.train_split = train_split
        self.scale_type = scale_type
        self.target_column = target_column
        self.data = converter(data)
        self.label_encode()
        self.split()
        self.data_train = self.scale(
            self.data_train, self.data_labels, self.target_column, self.scale_type
        )

    def label_encode(self):
        ret = label_encoder(self.data)
        self.data_encoded = ret["encoded_data"]
        self.data_labels = ret["encoded_labels"]

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
    def label_decode(data_encoded, data_labels):
        data_decoded = label_decoder(data_encoded, data_labels)
        return data_decoded

    @staticmethod
    def scale(
        input_data: np.ndarray, data_labels: list, target_column: int, scale_type: str
    ):

        # List of categorical vs numerical labels, without target_column
        input_predictor_labels = data_labels.copy()
        input_predictor_labels.pop(target_column)

        input_predictor_categorical_columns = [
            i for i, val in enumerate(input_predictor_labels) if val
        ]
        input_predictor_numerical_columns = [
            i for i, val in enumerate(input_predictor_labels) if not val
        ]

        # Remove target_column from input_data to match categorical_columns
        input_target = input_data[:, target_column]

        # Delete target column for scaling
        input_predictors_prep = np.delete(input_data, target_column, axis=1)

        # Delete categorical columns for scaling
        input_predictors = np.delete(
            input_predictors_prep, input_predictor_categorical_columns, axis=1
        )

        if target_column:
            predictors = scaler(input_data=input_predictors, scale_type=scale_type)
        else:
            predictors = input_predictors

        # Replace data with scaled data
        input_predictors_prep[:, input_predictor_numerical_columns] = predictors

        # Insert target data back in
        scaled_combined = np.insert(
            input_predictors_prep, target_column, input_target[np.newaxis], axis=1
        )
        return scaled_combined

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
