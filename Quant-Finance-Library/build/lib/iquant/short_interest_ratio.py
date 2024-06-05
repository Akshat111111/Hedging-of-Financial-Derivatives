import numpy as np

def short_interest_ratio(short_interest, outstanding_shares):
  """
  Calculates the short interest ratio.

  Args:
      short_interest: The number of shorted shares.
      outstanding_shares: The total number of outstanding shares.

  Returns:
      The short interest ratio as a float.
  """
  if outstanding_shares == 0:
    return np.inf  # Handle zero denominator
  return short_interest / outstanding_shares

# Example usage
short_interest = 1000000
outstanding_shares = 10000000
ratio = short_interest_ratio(short_interest, outstanding_shares)
print(f"Short interest ratio: {ratio}")
