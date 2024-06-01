import numpy as np

def max_drawdown(prices):
  """
  Calculates the maximum drawdown of a price series.

  Args:
      prices: A NumPy array of asset prices.

  Returns:
      The maximum drawdown as a float (percentage change).
  """
  peak = np.maximum.accumulate(prices)
  drawdown = (prices - peak) / peak * 100
  return drawdown.min()
