def kelly_criterion(win_rate, odds):
  """
  Calculates the Kelly Criterion for optimal f-planting.

  Args:
      win_rate: The win rate of the strategy (float between 0 and 1).
      odds: The average odds of winning trades (float, win amount / loss amount).

  Returns:
      The Kelly Criterion allocation ratio (float between 0 and 1).
  """
  if odds <= 1:
    return 0  # Handle cases where odds are not favorable
  return win_rate * (odds - 1)
