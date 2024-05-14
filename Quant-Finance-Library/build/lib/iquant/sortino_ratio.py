import numpy as np

def sortino_ratio(returns, risk_free_rate=0.0):
  """
  Calculates the Sortino ratio, which adjusts Sharpe ratio for downside risk.

  Args:
      returns (np.ndarray): A NumPy array of daily returns.
      risk_free_rate (float, optional): The risk-free rate of return.
                                        Defaults to 0.0.

  Returns:
      float: The Sortino ratio.
  """
  negative_returns = returns[returns < risk_free_rate]
  downside_risk = np.std(negative_returns)
  sortino = np.mean(returns - risk_free_rate) / downside_risk
  return sortino
