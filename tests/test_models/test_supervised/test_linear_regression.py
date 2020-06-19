import pytest
import numpy as np
from stoltzmaniac.models.supervised.linear_regression import LinearRegression


def test_linear_regression_model_input_parameters_passing(DATA_ARRAY_X, DATA_ARRAY_CLASSIFICATION_y, DATA_ARRAY_REGRESSION_y):
    # Check Defaults
    model_2d = LinearRegression(DATA_ARRAY_X, DATA_ARRAY_REGRESSION_y)
    assert (model_2d.X == DATA_ARRAY_X).all()
    assert (model_2d.y == DATA_ARRAY_REGRESSION_y).all()


def test_linear_regression_model_input_parameters_failing(DATA_ARRAY_X, DATA_ARRAY_CLASSIFICATION_y, DATA_ARRAY_REGRESSION_y):

    with pytest.raises(ValueError):
        LinearRegression(DATA_ARRAY_X, DATA_ARRAY_CLASSIFICATION_y)

    model_2d = LinearRegression(DATA_ARRAY_X, DATA_ARRAY_REGRESSION_y)
    with pytest.raises(ValueError):
        model_2d.predict(np.array([1.]))


def test_linear_regression_model_scale_values(DATA_ARRAY_X, DATA_ARRAY_REGRESSION_y):
    # Check default
    model_2d = LinearRegression(DATA_ARRAY_X, DATA_ARRAY_REGRESSION_y)
    assert (model_2d.X_train == np.array([[4., 18.], [1., 9.], [2., 13.]])).all()

    # Check scale_type passed properly
    scale_list = ["normalize", "standardize", "min_max", "scale"]
    for i in scale_list:
        model_2d = LinearRegression(DATA_ARRAY_X, DATA_ARRAY_REGRESSION_y, scale_type=i)
        assert model_2d.scale_type == i

def test_linear_regression_model_results_2D_no_scaling(DATA_ARRAY_X, DATA_ARRAY_REGRESSION_y):
    model_2d = LinearRegression(DATA_ARRAY_X, DATA_ARRAY_REGRESSION_y)

    # Check fit
    assert model_2d.betas[0] == pytest.approx(2, 0.001)
    assert model_2d.betas[1] == pytest.approx(0, 0.001)
    assert model_2d.betas[2] == pytest.approx(0, 0.001)

    # Check predictions
    predictions = model_2d.predict(np.array([[10, 11]]))
    assert predictions[0] == pytest.approx(20.0, 0.001)
