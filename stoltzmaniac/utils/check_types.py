import logging


def check_expected_type(actual, expected):
    """
    Checks expected types to ensure data integrity
    Parameters
    ----------
    actual
        any object

    expected
        expected data type


    Returns
    -------
    True (raises exception if types do not match)

    """

    if not type(actual) == expected:
        logging.warning(
            f"Input not of expected type.\n"
            f"Expected: {expected}\n"
            f"Actual: {type(actual)}"
        )
        return False
    else:
        return True
