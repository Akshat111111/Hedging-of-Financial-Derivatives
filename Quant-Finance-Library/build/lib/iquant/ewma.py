import numpy as np

def ewma_numpy(prices, window):
  """
  Calculates the Exponentially Weighted Moving Average (EWMA) using NumPy convolution.

  Args:
      prices: A NumPy array of asset prices.
      window: The window size for the EWMA.

  Returns:
      A NumPy array of the EWMA values.
  """
  alpha = 2 / (window + 1)  # Calculate alpha from window size
  weights = np.exp(np.linspace(-1, 0, window))  # Create exponential weights
  weights /= weights.sum()  # Normalize weights
  return np.convolve(prices, weights, mode='valid')

