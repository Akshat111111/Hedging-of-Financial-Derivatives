import numpy as np

def logarithmic_daily_return(prices):
  """
  Calculates the logarithmic daily return of a price series.

  Args:
      prices: A NumPy array of asset prices.

  Returns:
      A NumPy array of logarithmic daily returns.
  """
  returns = np.log(prices[1:]) - np.log(prices[:-1])
  return returns
