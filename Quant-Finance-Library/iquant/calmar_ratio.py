import annualized_sharpe_ratio

def calmar_ratio(daily_returns, annualization_factor=252):
  """
  Calculates the Calmar ratio of a series of daily returns.

  Args:
      daily_returns: A NumPy array of daily returns.
      annualization_factor: The number of trading days in a year (default 252).

  Returns:
      The Calmar ratio as a float.
  """
  return annualized_sharpe_ratio(daily_returns, annualization_factor=annualization_factor)
