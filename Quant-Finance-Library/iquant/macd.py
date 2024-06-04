import ewma

def macd(prices, fast_window, slow_window=12, signal_window=9):
  """
  Calculates the Moving Average Convergence Divergence (MACD) and signal line.

  Args:
      prices: A NumPy array of asset prices.
      fast_window: The window size for the fast EMA (default 12).
      slow_window: The window size for the slow EMA (default 26).
      signal_window: The window size for the signal line EMA (default 9).

  Returns:
      A tuple containing three NumPy arrays: MACD, signal line, MACD histogram.
  """
  ema_fast = ewma(prices, fast_window)
  ema_slow = ewma(prices, slow_window)
  macd = ema_fast - ema_slow
  signal_line = ewma(macd, signal_window)
  macd_hist = macd - signal_line
  return macd, signal_line, macd_hist
