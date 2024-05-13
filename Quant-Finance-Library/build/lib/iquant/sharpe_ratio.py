import numpy as np
import annualized_volatility

def sharpe_ratio(returns, risk_free_rate=0.0):
  """
  Calculates the Sharpe ratio, a measure of risk-adjusted return.

  Args:
      returns (np.ndarray): A NumPy array of daily returns.
      risk_free_rate (float, optional): The risk-free rate of return.
                                        Defaults to 0.0.

  Returns:
      float: The Sharpe ratio.
  """
  excess_returns = returns - risk_free_rate
  sharpe = np.mean(excess_returns) / annualized_volatility(returns)
  return sharpe
