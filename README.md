# stoltzmaniac  

A Python package to solve simple data science problems. 

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

----

Run tests  
`pytest`

Add packages (example with requests):

For development:
`poetry add -D requests`

For production:
`poetry add requests`

Build:
`poetry build`

Publish:
`poetry publish`