import beta

def capm(risk_free_rate, market_return, asset_return):
  """
  Calculates the expected return of an asset using the Capital Asset Pricing Model (CAPM).

  Args:
      risk_free_rate: The risk-free rate of return (float).
      market_return: The return of the market portfolio (float).
      asset_return: The return of the asset (float).

  Returns:
      The expected return of the asset based on CAPM (float).
  """
  beta = beta(asset_return, market_return)  # Calculate beta first
  expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
  return expected_return