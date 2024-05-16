import numpy as np

def information_ratio(portfolio_returns, benchmark_returns):
  """
  Calculates the Information Ratio of a portfolio.

  Args:
      portfolio_returns: A NumPy array of portfolio daily returns.
      benchmark_returns: A NumPy array of benchmark daily returns.

  Returns:
      The Information Ratio as a float.
  """
  excess_portfolio_return = portfolio_returns - np.mean(benchmark_returns)
  excess_benchmark_return = benchmark_returns - np.mean(benchmark_returns)
  tracking_error = np.std(excess_portfolio_return - excess_benchmark_return)
  if tracking_error == 0:
    return np.inf  # Handle zero denominator
  return excess_portfolio_return.mean() / tracking_error
