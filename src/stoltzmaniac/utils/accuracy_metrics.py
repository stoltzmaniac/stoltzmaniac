import numpy as np


def calculate_r_squared(y, y_hat):
    y_bar = np.sum(y) / len(y)
    ret = np.sum((y - y_hat) ** 2) / np.sum((y - y_bar) ** 2)
    return round(1 - ret, 5)
