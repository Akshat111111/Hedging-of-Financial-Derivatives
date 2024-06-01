import numpy as np

def annualized_volatility(returns, annualization_factor=252):
  """
  Calculates the annualized standard deviation (volatility) of returns.

  Args:
      returns (np.ndarray): A NumPy array of daily returns.
      annualization_factor (int, optional): The number of trading days in a year.
                                            Defaults to 252.

  Returns:
      float: The annualized standard deviation.
  """
  stddev = np.std(returns)
  annual_volatility = stddev * np.sqrt(annualization_factor)
  return annual_volatility
