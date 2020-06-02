import pytest
import numpy as np

from stoltzmaniac.data_handler.test_train_split_data import TrainTestSplitData


def test_test_train_data_input_types(DATA_ARRAY_3D):

    # Test not entering all parameters
    with pytest.raises(TypeError):
        TrainTestSplitData(DATA_ARRAY_3D)

    with pytest.raises(TypeError):
        TrainTestSplitData(DATA_ARRAY_3D, train_split=0.7)

    with pytest.raises(TypeError):
        TrainTestSplitData(DATA_ARRAY_3D, seed=10)

    # Test with bad seeds
    with pytest.raises(TypeError):
        TrainTestSplitData(DATA_ARRAY_3D, train_split=0.7, seed=0.1)

    with pytest.raises(ValueError):
        TrainTestSplitData(DATA_ARRAY_3D, train_split=0.7, seed=-2)

    # Test with bad train_split
    with pytest.raises(TypeError):
        TrainTestSplitData(DATA_ARRAY_3D, train_split="one", seed=1)

    with pytest.raises(ValueError):
        TrainTestSplitData(DATA_ARRAY_3D, train_split=1.2, seed=1)

    ttsd = TrainTestSplitData(DATA_ARRAY_3D, train_split=0.7, seed=1)
    assert ttsd.train_split == 0.7
    assert (ttsd.raw_data == DATA_ARRAY_3D).all()
    assert ttsd.seed == 1


def test_test_train_data_split_function(DATA_ARRAY_3D):
    ttsd = TrainTestSplitData(DATA_ARRAY_3D, train_split=0.5, seed=1)

    assert (ttsd.train_data == np.array([[4.0, 1.0, 8.0], [3.0, 9.0, 6.0]])).all()
    assert (ttsd.test_data == np.array([[1.0, 12.0, 2.0], [2.0, 3.0, 4.0]])).all()

    with pytest.raises(ValueError):
        TrainTestSplitData(np.array([[0, 1], [3, 5]]), train_split=0.9, seed=1)
