import numpy as np

def sortino_ratio_est_risk_free(daily_returns, risk_free_asset_returns):
  """
  Calculates the Sortino ratio with estimated risk-free return.

  Args:
      daily_returns: A NumPy array of daily returns for the asset.
      risk_free_asset_returns: A NumPy array of daily returns for the risk-free asset.

  Returns:
      The Sortino ratio as a float.
  """
  risk_free_rate = np.mean(risk_free_asset_returns)
  downside_risk = daily_returns[daily_returns < risk_free_rate].std()
  if downside_risk == 0:
    return np.inf  # Handle zero denominator
  excess_returns = daily_returns - risk_free_rate
  return excess_returns.mean() / downside_risk
