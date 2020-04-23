import pytest
import numpy as np
from stoltzmaniac.models.supervised.linear_regression import LinearRegression


def test_linear_regression_model_input_parameters_passing(DATA_ARRAY_2D):
    # Check Defaults
    model_2d = LinearRegression(DATA_ARRAY_2D)
    assert model_2d.seed == 123
    assert model_2d.scale_type is None

    # Check scale_type passed properly
    scale_list = ["normalize", "standardize", "min_max", "scale"]
    for i in scale_list:
        model_2d = LinearRegression(DATA_ARRAY_2D, scale_type=i)
        assert model_2d.scale_type == i

    model_2d = LinearRegression(DATA_ARRAY_2D, seed=10)
    assert model_2d.seed == 10


def test_linear_regression_model_input_parameters_failing(DATA_ARRAY_2D):

    with pytest.raises(TypeError):
        LinearRegression(DATA_ARRAY_2D, scale_type=4)

    with pytest.raises(TypeError):
        LinearRegression(DATA_ARRAY_2D, seed=None)

    with pytest.raises(TypeError):
        LinearRegression(DATA_ARRAY_2D, train_split=0)

    with pytest.raises(ValueError):
        LinearRegression(DATA_ARRAY_2D, train_split=-0.2)

    with pytest.raises(ValueError):
        LinearRegression(DATA_ARRAY_2D, train_split=-1.3)

    with pytest.raises(TypeError):
        LinearRegression(DATA_ARRAY_2D, train_split="hi")


def test_linear_regression_model_scale_values(DATA_ARRAY_2D):
    # Check default
    model_2d = LinearRegression(DATA_ARRAY_2D)
    assert all(model_2d.split_data.train_data == np.array([[4.0], [1.0], [2.0]]))

    # Check from scale type list
    model_2d = LinearRegression(DATA_ARRAY_2D, scale_type=None)
    assert all(model_2d.split_data.train_data == np.array([[4.0], [1.0], [2.0]]))

    # Check scale_type passed properly
    scale_list = ["normalize", "standardize", "min_max", "scale"]
    for i in scale_list:
        model_2d = LinearRegression(DATA_ARRAY_2D, scale_type=i)
        assert model_2d.scale_type == i

    model_2d = LinearRegression(DATA_ARRAY_2D, seed=10)
    assert model_2d.seed == 10

    model_2d = LinearRegression(DATA_ARRAY_2D, scale_type="normalize")
    assert model_2d.split_data.train_data[0] == pytest.approx(0.55555, 0.001)
    assert model_2d.split_data.train_data[1] == pytest.approx(-0.44444, 0.001)
    assert model_2d.split_data.train_data[2] == pytest.approx(-0.11111, 0.001)

    model_2d = LinearRegression(DATA_ARRAY_2D, scale_type="scale")
    assert model_2d.split_data.train_data[0] == pytest.approx(1.6666, 0.001)
    assert model_2d.split_data.train_data[1] == pytest.approx(-1.3333, 0.001)
    assert model_2d.split_data.train_data[2] == pytest.approx(-0.3333, 0.001)

    model_2d = LinearRegression(DATA_ARRAY_2D, scale_type="min_max")
    assert model_2d.split_data.train_data[0] == pytest.approx(1.8, 0.001)
    assert model_2d.split_data.train_data[1] == pytest.approx(0, 0.001)
    assert model_2d.split_data.train_data[2] == pytest.approx(0.6, 0.001)

    model_2d = LinearRegression(DATA_ARRAY_2D, scale_type="standardize")
    assert model_2d.split_data.train_data[0] == pytest.approx(1.3363, 0.001)
    assert model_2d.split_data.train_data[1] == pytest.approx(-1.0690, 0.001)
    assert model_2d.split_data.train_data[2] == pytest.approx(-0.2673, 0.001)


def test_linear_regression_model_results_2D_no_scaling(DATA_ARRAY_2D):
    model_2d = LinearRegression(DATA_ARRAY_2D)

    # Check fit
    assert model_2d.betas[0] == 2.0
    assert model_2d.betas[1] == pytest.approx(0, 0.001)

    # Check predictions
    predictions = model_2d.predict(np.array([[10], [11]]))
    assert predictions[0] == pytest.approx(20.0, 0.001)
    assert predictions[1] == pytest.approx(22.0, 0.001)


def test_linear_regression_model_results_3D_no_scaling(DATA_ARRAY_3D):
    model_3d = LinearRegression(DATA_ARRAY_3D)

    # Check fit
    assert model_3d.betas[0] == pytest.approx(2, 0.001)
    assert model_3d.betas[1] == pytest.approx(0, 0.001)

    # Check predictions
    predictions = model_3d.predict(np.array([[10, 12], [11, 13]]))
    assert predictions[0] == pytest.approx(20.0, 0.001)
    assert predictions[1] == pytest.approx(22.0, 0.001)
