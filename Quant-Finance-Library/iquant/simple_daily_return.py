import numpy as np

def simple_daily_return(prices):
  """
  Calculates the simple daily return of an asset price series.

  Args:
      prices (np.ndarray): A NumPy array of asset prices.

  Returns:
      np.ndarray: A NumPy array of daily returns.
  """
  returns = np.diff(prices) / prices[:-1]
  return returns
