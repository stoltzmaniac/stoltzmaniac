import pytest
import numpy as np

from stoltzmaniac.data_handler.scale_data import ScaleData


def test_scale_data_input_parameters(DATA_ARRAY_3D):

    with pytest.raises(TypeError):
        ScaleData(DATA_ARRAY_3D)

    with pytest.raises(TypeError):
        ScaleData(DATA_ARRAY_3D, scale_type=2)


def test_scale_data_results(DATA_ARRAY_2D):
    sd = ScaleData(DATA_ARRAY_2D, scale_type=None)
    assert sd.scale_type is None
    assert (sd.original_data == DATA_ARRAY_2D).all()

    scale_list = ["normalize", "standardize", "min_max", "scale"]
    for i in scale_list:
        sd = ScaleData(DATA_ARRAY_2D, scale_type=i)
        assert sd.scale_type == i

    x_2d = np.array([[1., 1.], [2., 2.], [3., 3.]])
    sd = ScaleData(x_2d, scale_type=None)
    comparison = sd.array_mean == np.mean(x_2d, axis=0)
    assert comparison.all()
    comparison = sd.array_max == np.max(x_2d, axis=0)
    assert comparison.all()
    comparison = sd.array_min == np.min(x_2d, axis=0)
    assert comparison.all()
    comparison = sd.array_std == np.std(x_2d, axis=0)
    assert comparison.all()

    model_2d = ScaleData(DATA_ARRAY_2D, scale_type="normalize")
    assert (
        np.round(model_2d.original_scaled_data, 2)
        == np.round(np.array([[-0.5], [-0.16666667], [0.16666667], [0.5]]), 2)
    ).all()

    model_2d = ScaleData(DATA_ARRAY_2D, scale_type="min_max")
    assert (
        np.round(model_2d.original_scaled_data, 2)
        == np.round(np.array([[0.0], [0.66666667], [1.33333333], [2.0]]), 2)
    ).all()

    model_2d = ScaleData(DATA_ARRAY_2D, scale_type="standardize")
    assert (
        np.round(model_2d.original_scaled_data, 2)
        == np.round(
            np.array([[-1.34164079], [-0.4472136], [0.4472136], [1.34164079]]), 2
        )
    ).all()

    with pytest.raises(ValueError):
        tmp = ScaleData(DATA_ARRAY_2D, scale_type="not_in_my_list")
        tmp.scale(DATA_ARRAY_2D)
