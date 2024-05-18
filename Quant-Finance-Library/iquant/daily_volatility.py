def daily_volatility(daily_returns):
  """
  Calculates the daily standard deviation (volatility) of a series of returns.

  Args:
      daily_returns: A NumPy array of daily returns.

  Returns:
      The daily standard deviation as a float.
  """
  return daily_returns.std()
