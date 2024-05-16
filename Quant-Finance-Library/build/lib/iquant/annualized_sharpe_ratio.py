import daily_volatility

def annualized_sharpe_ratio(daily_returns, risk_free_rate=0.0, annualization_factor=252):
  """
  Calculates the annualized Sharpe ratio of a series of daily returns.

  Args:
      daily_returns: A NumPy array of daily returns.
      risk_free_rate: The risk-free rate (default 0.0).
      annualization_factor: The number of trading days in a year (default 252).

  Returns:
      The annualized Sharpe ratio as a float.
  """
  excess_returns = daily_returns - risk_free_rate
  return (excess_returns.mean() * annualization_factor**0.5) / daily_volatility(daily_returns)
