import numpy as np
from scipy.stats import norm

def black_scholes(spot_price, strike_price, time_to_maturity, volatility, risk_free_rate, option_type='call'):
  """
  Calculates the Black-Scholes option price and greeks (delta, gamma, vega, theta, rho).

  Args:
      spot_price: The current spot price of the underlying asset.
      strike_price: The strike price of the option.
      time_to_maturity: The time to maturity of the option in years.
      volatility: The annualized volatility of the underlying asset.
      risk_free_rate: The annualized risk-free rate.
      option_type: The option type ('call' or 'put', default 'call').

  Returns:
      A dictionary containing the option price and greeks.
  """
  d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
  d2 = d1 - volatility * np.sqrt(time_to_maturity)

  if option_type == 'call':
    price = spot_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
  else:
    price = strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)

  # Calculate Greeks
  delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(-d1)
  gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_maturity))
  vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_maturity)
  theta = -spot_price * volatility * norm.pdf(d1) / (2 * np.sqrt(time_to_maturity)) - risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2) if option_type == 'call' else -spot_price * volatility * norm.pdf(d1) / (2 * np.sqrt(time_to_maturity)) + risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)
  rho = strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2) if option_type == 'call' else -strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)

  return {
      'Price': price,
      'Delta': delta,
      'Gamma': gamma,
      'Vega': vega,
      'Theta': theta,
      'Rho': rho
  }

