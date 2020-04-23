# stoltzmaniac  

A Python package to solve simple data science problems. 

![travisci](https://travis-ci.com/stoltzmaniac/stoltzmaniac.svg?branch=master)
![codecov](https://codecov.io/gh/stoltzmaniac/stoltzmaniac/branch/master/graph/badge.svg)

This is a rudimentary library for machine learning with 3 concepts in mind. This library should be:
  - easy-to-use
  - easy-to-read
  - easy-to-interpret

Optimization for efficiency of compute, latency, and memory will not be a priority.

The only external package we will use for modeling will be `numpy`. Others required are simply to get data into `numpy.ndarray` format.

This is my first package, so help out and don't hold back on putting in PR's. Thank you!

Installing:
```bash
pip install stoltzmaniac
```

Example of Linear Regression:
```python
import numpy as np
from stoltzmaniac.models.supervised.linear_regression import LinearRegression

my_array = np.array([[1, 12, 2], [2, 3, 4], [3, 9, 6], [4, 1, 8]], dtype=np.float)
model = LinearRegression(my_array, train_split=0.7, scale_type="normalize")

model.predict(np.array([[10, 12], [50, 4]]))
# >> array([ 20., 100.])
```

----

Run tests  
`pytest`

Add packages (example with requests):

For development:
`poetry add -D requests`

For production:
`poetry add requests`

Changes to version must be made in:
```shell script
stoltzmaniac/__init__.py
tests/test_stoltzmaniac.py
pyproject.toml
```

Automatically generate docs (write appropriate lines into `index.rst`)
`make html`

Build:
`poetry build`

Publish:
`poetry publish`