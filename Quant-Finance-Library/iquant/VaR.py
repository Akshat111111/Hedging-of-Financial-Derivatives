import numpy as np

def historical_var(returns, confidence_level=0.95):
  """
  Calculates the Historical VaR based on the empirical distribution of returns.

  Args:
      returns (np.ndarray): A NumPy array of daily returns.
      confidence_level (float, optional): The confidence level for VaR.
                                           Defaults to 0.95.

  Returns:
      float: The Historical VaR value.
  """
  quantile = 1 - confidence_level
  var = np.percentile(returns, quantile * 100)
  return var
