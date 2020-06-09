from collections import Counter
import numpy as np

from stoltzmaniac.data_handler.base import ArrayData
from stoltzmaniac.data_handler.clean_data import CleanData
from stoltzmaniac.data_handler.scale_data import ScaleData
from stoltzmaniac.data_handler.test_train_split_data import TrainTestSplitData
from stoltzmaniac.utils.distance_functions import euclidian_distance


class KNearestNeighbors:
    def __init__(
        self,
        input_data,
        n_neighbors: int = 5,
        train_split=0.7,
        scale_type=None,
        seed=123,
    ):
        """
        Create a KNN model
        Parameters
        ----------
        input_data: can be of any type easily converted to np.ndarray
        n_neighbors: utilize n closest number of neighbors
        train_split: float described in TrainTestSplitData model
        scale_type: str described in ScaleData model
        seed: flat described in TrainTestSplitData model
        """
        self.raw_data = input_data
        self.n_neighbors = n_neighbors
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

        # Get number of classes in train data
        self.n_classes = len(np.unique(self.scaler.y_data))

    @staticmethod
    def calculate_distance(a, b, distance_type):
        if distance_type == "euclidean":
            return euclidian_distance(a, b)
        else:
            raise ValueError(f"distance is not recognized, currently {distance_type}")

    def predict(self, new_data: np.ndarray, distance_type: str = "euclidean"):
        all_distances = []
        for new_row in new_data:
            distances = []
            for comp_row, group in zip(self.scaler.x_data, self.scaler.y_data):
                distance = self.calculate_distance(comp_row, new_row, distance_type)
                distances.append([distance, group])
            votes = [i[1] for i in sorted(distances)[: self.n_neighbors]]
            popular_vote = Counter(votes).most_common(1)[0][0]
            all_distances.append(popular_vote)
        return all_distances


# my_array = np.array([[1., 12., 2., 10], [2., 3., 4., 12], [3., 9., 6., 14], [4., 1., 8., 16]])
# knn = KNearestNeighbors(my_array)
# data_to_predict = np.array([[1., 12., 2], [2.1, 3.1, 4.1]])
# data_to_predict = np.array([[1., 12., 2.], [2., 3., 4.], [3., 9., 6.], [4., 1., 8.]])
# a = knn.predict(data_to_predict)
# print(knn.scaler.x_data)
# print(knn.scaler.y_data)
# print(a)
# knn.score(data_to_predict, 'euclidean', 'uniform')
# data_to_predict = np.array([[1., 13.], [2., 4.], [3., 10.], [4., 2.]])
