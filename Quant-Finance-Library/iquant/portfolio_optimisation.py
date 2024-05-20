import numpy as np
from scipy.optimize import minimize


def portfolio_optimization(expected_returns, covariance_matrix, target_risk):
  """
  Optimizes portfolio weights for maximum expected return with target risk.

  Args:
      expected_returns: A NumPy array of expected returns for each asset.
      covariance_matrix: A NumPy array representing the covariance matrix of asset returns.
      target_risk: The target risk level for the portfolio.

  Returns:
      A NumPy array of optimal portfolio weights, a float representing the achieved portfolio risk, and a float representing the achieved portfolio expected return.
  """

  num_assets = len(expected_returns)

  # Define objective function (minimize negative expected return)
  def negative_expected_return(weights):
    return -sum(weights * expected_returns)

  # Define constraint (sum of weights equals 1)
  def weights_sum_to_one(weights):
    return sum(weights) - 1

  # Define initial guess for weights (equal weights)
  initial_weights = np.ones(num_assets) / num_assets

  # Define bounds (0 to 1 for each weight)
  bounds = tuple((0, 1) for _ in range(num_assets))

  # Set constraints
  constraints = ({'type': 'eq', 'fun': weights_sum_to_one})

  # Solve the optimization problem
  sol = minimize(negative_expected_return, initial_weights, method='SLSQP',
                  bounds=bounds, constraints=constraints)

  # Extract weights, portfolio risk, and portfolio expected return
  optimal_weights = sol.x
  portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
  portfolio_expected_return = sum(optimal_weights * expected_returns)

  return optimal_weights, portfolio_risk, portfolio_expected_return
