import numpy as np
from scipy.optimize import minimize

def portfolio_optimization(expected_returns, covariance_matrix, target_risk):
    """
    Optimize portfolio weights for maximum expected return with target risk.

    Args:
        expected_returns (array-like): Expected returns for each asset.
        covariance_matrix (array-like): Covariance matrix of asset returns.
        target_risk (float): Target risk level for the portfolio.

    Returns:
        dict: A dictionary containing the optimal portfolio weights, achieved portfolio risk, and achieved portfolio expected return.
    """

    # Convert inputs to NumPy arrays
    expected_returns = np.array(expected_returns)
    covariance_matrix = np.array(covariance_matrix)

    # Parameter validation
    if len(expected_returns) != len(covariance_matrix):
        raise ValueError("Length of expected returns and covariance matrix must be the same")

    num_assets = len(expected_returns)

    # Define objective function (minimize negative expected return)
    def negative_expected_return(weights):
        return -np.sum(weights * expected_returns)

    # Define constraint (sum of weights equals 1)
    def weights_sum_to_one(weights):
        return np.sum(weights) - 1

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
    portfolio_expected_return = np.sum(optimal_weights * expected_returns)

    # Return results as a dictionary
    results = {
        'optimal_weights': optimal_weights,
        'portfolio_risk': portfolio_risk,
        'portfolio_expected_return': portfolio_expected_return
    }

    return results

if __name__ == "__main__":
    # Example data
    expected_returns = [0.05, 0.07, 0.06]
    covariance_matrix = [[0.1, 0.05, 0.03],
                         [0.05, 0.12, 0.07],
                         [0.03, 0.07, 0.15]]
    target_risk = 0.1  # Example target risk level

    # Call the function
    results = portfolio_optimization(expected_returns, covariance_matrix, target_risk)

    # Print the results
    print("Optimal Weights:", results['optimal_weights'])
    print("Portfolio Risk:", results['portfolio_risk'])
    print("Portfolio Expected Return:", results['portfolio_expected_return'])
