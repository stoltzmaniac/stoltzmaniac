from stoltzmaniac.data_handler.clean_data import CleanData


def test_clean_data(DATA_ARRAY_3D):

    cd = CleanData(DATA_ARRAY_3D)
    assert (cd.clean_data == cd.raw_data).all()
    assert (cd.clean_data == DATA_ARRAY_3D).all()
