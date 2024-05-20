import numpy as np

def simple_moving_average(prices, window):
  """
  Calculates the Simple Moving Average (SMA) of a price series.

  Args:
      prices: A NumPy array of asset prices.
      window: The window size for the SMA.

  Returns:
      A NumPy array of the SMA values.
  """
  return np.convolve(prices, np.ones(window) / window, mode='valid')
