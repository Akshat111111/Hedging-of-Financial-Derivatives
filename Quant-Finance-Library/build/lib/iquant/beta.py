import numpy as np

def beta(returns_x, returns_y):
  """
  Calculates the beta of asset X relative to asset Y.

  Args:
      returns_x: A NumPy array of returns for asset X.
      returns_y: A NumPy array of returns for asset Y.

  Returns:
      The beta coefficient as a float.
  """
  covariance = np.cov(returns_x, returns_y)[0, 1]
  variance_y = np.var(returns_y)
  return covariance / variance_y
