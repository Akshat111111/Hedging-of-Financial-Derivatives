import numpy as np
import simple_moving_average
import daily_volatility

def bollinger_bands(prices, window=20, std=2):
  """
  Calculates the Bollinger Bands (upper and lower bands) of a price series.

  Args:
      prices: A NumPy array of asset prices.
      window: The window size for the moving average (default 20).
      std: The number of standard deviations for the bands (default 2).

  Returns:
      A tuple containing three NumPy arrays: moving average, upper band, lower band.
  """
  moving_average = simple_moving_average(prices, window)
  stddev = daily_volatility(prices[window:])
  upper_band = moving_average + std * stddev
  lower_band = moving_average - std * stddev
  return moving_average, upper_band, lower_band
