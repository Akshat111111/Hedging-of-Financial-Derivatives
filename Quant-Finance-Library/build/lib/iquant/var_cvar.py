import numpy as np

def var_cvar(daily_returns, confidence_level=0.95):
  """
  Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR).

  Args:
      daily_returns: A NumPy array of daily returns.
      confidence_level: The confidence level for VaR (default 0.95).

  Returns:
      A tuple containing VaR and CVaR values as floats.
  """
  percentiles = np.percentile(daily_returns, [100 - confidence_level * 100])
  var_value = percentiles[0]

  # Calculate CVaR (average loss within VaR confidence level)
  exceedances = daily_returns[daily_returns < var_value]
  cvar_value = np.mean(exceedances)

  return var_value, cvar_value
