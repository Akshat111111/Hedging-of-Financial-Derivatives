import numpy as np

def rsi(prices, window=14):
  """
  Calculates the Relative Strength Index (RSI) of a price series.

  Args:
      prices: A NumPy array of asset prices.
      window: The window size for the RSI (default 14).

  Returns:
      A NumPy array of the RSI values.
  """
  delta = prices[1:] - prices[:-1]
  up, down = np.where(delta >= 0, delta, 0), np.where(delta < 0, abs(delta), 0)
  ema_up = np.convolve(up, np.ones(window) / window, mode='valid')
  ema_down = np.convolve(down, np.ones(window) / window, mode='valid')
  rs = ema_up / ema_down
  rsi = 100 - 100 / (1 + rs)
  return rsi
