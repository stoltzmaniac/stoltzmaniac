import pytest
import numpy as np
from stoltzmaniac.data_handler.base import BaseData, UnsupervisedData, SupervisedData, RegressionData, ClassificationData


def test_base_data_input_types(DATA_ARRAY_3D):
    """
    Ensure errors are thrown for non ndarray data and that proper arrays are returned
    """
    with pytest.raises(ValueError):
        BaseData({"a": 1, "b": 3})

    with pytest.raises(ValueError):
        BaseData({1, 5, 3, 2})

    with pytest.raises(ValueError):
        BaseData([1, 2, 4])

    with pytest.raises(ValueError):
        BaseData(np.array([['a'], ['b']]))

    array_data = BaseData(DATA_ARRAY_3D)
    assert (array_data.data == DATA_ARRAY_3D).all()

    array_data = BaseData(np.array([1.0, 2.0, 4.0]))
    assert (array_data.data == np.array([[1.0], [2.0], [4.0]])).all()


def test_base_data_description(DATA_ARRAY_3D):
    array_data = BaseData(DATA_ARRAY_3D)

    exp = (
        f"Data specs:\n"
        f"type: {type(array_data.data)}\n"
        f"shape: {array_data.data.shape}\n"
        f"ndim: {array_data.data.ndim}"
    )

    assert print(exp) == print(array_data)

def test_unsupervised_data(DATA_ARRAY_3D, DATA_ARRAY_2D):
    """
    Ensuring data comes back as proper array
    """
    comparison = DATA_ARRAY_3D == UnsupervisedData(DATA_ARRAY_3D).X
    assert comparison.all()

    comparison = DATA_ARRAY_2D == UnsupervisedData(DATA_ARRAY_2D).X
    assert comparison.all()

    comparison = np.array([[1], [2], [3]]) == UnsupervisedData(np.array([[1], [2], [3]])).X
    assert comparison.all()

    comparison = np.array([[1], [2], [3]]) == UnsupervisedData(np.array([1, 2, 3])).X
    assert comparison.all()

def test_supervised_data(DATA_ARRAY_3D, DATA_ARRAY_2D):
    """
    Ensuring data comes back as proper array and test value errors
    """

    x_1d = np.array([[1], [2], [3]])
    x_2d = np.array([[1, 1], [2, 2], [3, 3]])
    y_1d = np.array([[1], [2], [3]])
    y_1d_single = np.array([1, 2, 3])

    sd = SupervisedData(x_1d, y_1d)
    comparison = sd.X == x_1d
    assert comparison.all()

    comparison = sd.y == y_1d
    assert comparison.all()

    sd = SupervisedData(x_1d, y_1d_single)
    comparison = sd.y == y_1d
    assert comparison.all()

    sd = SupervisedData(x_2d, y_1d)
    comparison = sd.X == x_2d
    assert comparison.all()

    comparison = sd.y == y_1d
    assert comparison.all()

    with pytest.raises(ValueError):
        SupervisedData(x_1d, x_2d)

    with pytest.raises(AttributeError):
        SupervisedData(x_2d, np.array([[1], [2]]))

def test_regression_data(DATA_ARRAY_3D, DATA_ARRAY_2D):
    """
    Ensuring data comes back as proper array and test value errors
    """
    x_1d = np.array([[1.], [2.], [3.]])
    x_2d = np.array([[1., 1.], [2., 2.], [3., 3.]])
    y_1d = np.array([[1.], [2.], [3.]])
    y_1d_single = np.array([1., 2., 3.])
    y_1d_int = np.array([[1], [2], [3]])

    sd = RegressionData(x_1d, y_1d)
    comparison = sd.X == x_1d
    assert comparison.all()

    comparison = sd.y == y_1d
    assert comparison.all()

    sd = RegressionData(x_1d, y_1d_single)
    comparison = sd.y == y_1d
    assert comparison.all()

    sd = RegressionData(x_2d, y_1d)
    comparison = sd.X == x_2d
    assert comparison.all()

    comparison = sd.y == y_1d
    assert comparison.all()

    with pytest.raises(ValueError):
        RegressionData(x_1d, x_2d)

    with pytest.raises(AttributeError):
        RegressionData(x_2d, np.array([[1], [2]]))

    with pytest.raises(ValueError):
        RegressionData(x_1d, y_1d_int)


def test_classification_data(DATA_ARRAY_3D, DATA_ARRAY_2D):
    """
    Ensuring data comes back as proper array and test value errors
    """
    x_1d = np.array([[1.], [2.], [3.]])
    x_2d = np.array([[1., 1.], [2., 2.], [3., 3.]])
    y_1d = np.array([[1.], [2.], [3.]])
    y_1d_single = np.array([1, 2, 3])
    y_1d_int = np.array([[1], [2], [3]])

    sd = ClassificationData(x_1d, y_1d_int)
    comparison = sd.X == x_1d
    assert comparison.all()

    comparison = sd.y == y_1d_int
    assert comparison.all()

    sd = ClassificationData(x_1d, y_1d_single)
    comparison = sd.y == y_1d_int
    assert comparison.all()

    sd = ClassificationData(x_2d, y_1d_int)
    comparison = sd.X == x_2d
    assert comparison.all()

    comparison = sd.y == y_1d_int
    assert comparison.all()

    with pytest.raises(ValueError):
        ClassificationData(x_1d, x_2d)

    with pytest.raises(AttributeError):
        ClassificationData(x_2d, np.array([[1], [2]]))

    with pytest.raises(ValueError):
        ClassificationData(x_1d, y_1d)