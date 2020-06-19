from collections import Counter
import numpy as np

from stoltzmaniac.data_handler.base import BaseData, ClassificationData
from stoltzmaniac.data_handler.scale_data import ScaleData
from stoltzmaniac.data_handler.test_train_split_data import TrainTestSplitData
from stoltzmaniac.utils.distance_functions import euclidian_distance


class KNearestNeighbors:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_neighbors: int = 5,
        train_split=0.7,
        scale_type=None,
        seed=123,
    ):
        """
        Create a KNN model with scale type and k neighbors
        Parameters
        ----------
        X: array of predictor variables
        y: array of target variable
        n_neighbors: utilize n closest number of neighbors
        train_split: float described in TrainTestSplitData model
        scale_type: str described in ScaleData model
        seed: flat described in TrainTestSplitData model
        """
        self.n_neighbors = n_neighbors
        self.scale_type = scale_type

        self.data = ClassificationData(X=X, y=y)
        self.X = self.data.X
        self.Y = self.data.y

        X_split = TrainTestSplitData(X, train_split=train_split, seed=seed)
        y_split = TrainTestSplitData(y, train_split=train_split, seed=seed)
        self.X_train = X_split.train_data
        self.X_test = X_split.test_data
        self.y_train = y_split.train_data
        self.y_test = y_split.test_data

        # Set scaling parameters, assigns fixed scaling parameters
        self.scaler = ScaleData(self.X_train, scale_type=self.scale_type)

        # Get number of classes in train data
        self.n_classes = len(np.unique(self.y_train))

    @staticmethod
    def calculate_distance(a, b, distance_type):
        if distance_type == "euclidean":
            return euclidian_distance(a, b)
        else:
            raise ValueError(
                f"distance_type is not recognized, currently {distance_type}"
            )

    def predict(self, data: np.ndarray, distance_type: str = "euclidean"):

        new_data = self.scaler.scale(BaseData(data).data)
        if new_data.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"predict data must be the same number of columns as the original X data. # of Columns: Original X = {self.X.shape[1]}, current data = {new_data.shape[1]}"
            )

        all_distances = []
        for new_row in new_data:
            distances = []
            for comp_row, group in zip(self.scaler.original_scaled_data, self.y_train):
                distance = self.calculate_distance(comp_row, new_row, distance_type)
                distances.append([distance, int(group)])
            votes = [i[1] for i in sorted(distances)[: self.n_neighbors]]
            popular_vote = Counter(votes).most_common(1)[0][0]
            all_distances.append(popular_vote)
        return all_distances
