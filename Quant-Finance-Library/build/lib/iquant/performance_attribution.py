import numpy as np

def performance_attribution(portfolio_returns, benchmark_returns):
  """
  Calculates a basic performance attribution (active return vs. benchmark return).

  Args:
      portfolio_returns: A NumPy array of portfolio daily returns.
      benchmark_returns: A NumPy array of benchmark daily returns.

  Returns:
      A tuple containing two NumPy arrays: active return and tracking error.
  """
  active_return = portfolio_returns - benchmark_returns
  tracking_error = np.std(portfolio_returns - benchmark_returns)
  return active_return, tracking_error
