import numpy as np
from scipy.optimize import minimize

def portfolio_optimization(expected_returns: np.ndarray, covariance_matrix: np.ndarray, target_risk: float, allow_short: bool = False, include_risk_free: bool = False, risk_free_rate: float = 0.0) -> dict:
    """
    Optimize portfolio weights for maximum expected return with target risk.

    Args:
        expected_returns (np.ndarray): Expected returns for each asset.
        covariance_matrix (np.ndarray): Covariance matrix of asset returns.
        target_risk (float): Target risk level for the portfolio.
        allow_short (bool): Allow short selling (negative weights) if True.
        include_risk_free (bool): Include a risk-free asset if True.
        risk_free_rate (float): Return of the risk-free asset.

    Returns:
        dict: A dictionary containing the optimal portfolio weights, achieved portfolio risk, achieved portfolio expected return, and Sharpe ratio.
    """

    # Convert inputs to NumPy arrays
    expected_returns = np.array(expected_returns)
    covariance_matrix = np.array(covariance_matrix)

    # Parameter validation
    num_assets = len(expected_returns)
    if num_assets != covariance_matrix.shape[0] or num_assets != covariance_matrix.shape[1]:
        raise ValueError("Expected returns and covariance matrix dimensions must match")

    # If including risk-free asset, extend the arrays
    if include_risk_free:
        expected_returns = np.append(expected_returns, risk_free_rate)
        extra_row = np.zeros((1, num_assets))
        extra_col = np.zeros((num_assets + 1, 1))
        covariance_matrix = np.vstack((covariance_matrix, extra_row))
        covariance_matrix = np.hstack((covariance_matrix, extra_col))

    num_assets = len(expected_returns)

    # Define objective function (minimize negative expected return)
    def negative_expected_return(weights):
        return -np.sum(weights * expected_returns)

    # Define constraint (sum of weights equals 1)
    def weights_sum_to_one(weights):
        return np.sum(weights) - 1

    # Define risk constraint
    def risk_constraint(weights):
        return target_risk - np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    # Define initial guess for weights (equal weights)
    initial_weights = np.ones(num_assets) / num_assets

    # Define bounds
    if allow_short:
        bounds = tuple((-1, 1) for _ in range(num_assets))
    else:
        bounds = tuple((0, 1) for _ in range(num_assets))

    # Set constraints
    constraints = [{'type': 'eq', 'fun': weights_sum_to_one},
                   {'type': 'ineq', 'fun': risk_constraint}]

    # Solve the optimization problem
    sol = minimize(negative_expected_return, initial_weights, method='SLSQP',
                   bounds=bounds, constraints=constraints)

    if not sol.success:
        raise ValueError("Optimization failed: " + sol.message)

    # Extract weights, portfolio risk, and portfolio expected return
    optimal_weights = sol.x
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
    portfolio_expected_return = np.sum(optimal_weights * expected_returns)

    # Calculate the Sharpe ratio
    sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_risk if portfolio_risk != 0 else np.nan

    # Return results as a dictionary
    results = {
        'optimal_weights': optimal_weights,
        'portfolio_risk': portfolio_risk,
        'portfolio_expected_return': portfolio_expected_return,
        'sharpe_ratio': sharpe_ratio
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
    results = portfolio_optimization(expected_returns, covariance_matrix, target_risk, allow_short=True, include_risk_free=True, risk_free_rate=0.02)

    # Print the results
    print("Optimal Weights:", results['optimal_weights'])
    print("Portfolio Risk:", results['portfolio_risk'])
    print("Portfolio Expected Return:", results['portfolio_expected_return'])
    print("Sharpe Ratio:", results['sharpe_ratio'])
