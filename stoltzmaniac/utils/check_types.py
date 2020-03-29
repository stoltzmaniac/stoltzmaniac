def check_expected_type(actual, expected):
    """
    Checks expected types to ensure data integrity
    :param actual: any object
    :param expected: expected data type
    :return: None (raises exception if types do not match)
    """
    if not type(actual) == expected:
        raise TypeError(
            f"Input not of expected type.\n"
            f"Expected: {expected}\n"
            f"Actual: {type(actual)}"
        )
