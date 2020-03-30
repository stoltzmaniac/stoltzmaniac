class Base:
    def __init__(self, input_data):
        """
        Base Class for data input, houses raw input
        Parameters
        ----------
        input_data
        """
        # Automatically throws TypeError if input_data not given
        self.input_data = input_data
