import numpy as np
from scipy.stats import norm

def monte_carlo_simulation(spot_price, drift, volatility, simulations, time_horizon):
  """
  Simulates random price paths for an asset using geometric Brownian motion.

  Args:
      spot_price: The initial spot price of the asset.
      drift: The expected return (drift) of the asset (annualized).
      volatility: The annualized volatility of the asset.
      simulations: The number of simulations to run.
      time_horizon: The time horizon for the simulation (number of periods).

  Returns:
      A NumPy array of simulated price paths.
  """
  dt = time_horizon / 252  # Convert time horizon to days (assuming 252 trading days/year)
  drift_term = (drift - 0.5 * volatility**2) * dt
  volatility_term = volatility * np.sqrt(dt) * norm.rvs(size=(simulations, time_horizon))

  price_paths = np.empty((simulations, time_horizon + 1))
  price_paths[:, 0] = spot_price
  for i in range(1, time_horizon + 1):
    price_paths[:, i] = price_paths[:, i-1] * np.exp(drift_term + volatility_term[:, i-1])

  return price_paths

