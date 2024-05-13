import numpy as np

def credit_rating_migration(transition_matrix, initial_ratings):
  """
  Estimates credit rating migration probabilities (basic example).

  Args:
      transition_matrix: A NumPy array representing the transition matrix (rows: origin ratings, columns: destination ratings).
      initial_ratings: A NumPy array representing the initial credit rating distribution (e.g., proportions).

  Returns:
      A NumPy array representing the credit rating distribution after one period.
  """
  rating_probs = np.dot(initial_ratings, transition_matrix)
  return rating_probs

# Example usage (assuming a simplified 3-rating system)
transition_matrix = np.array([[0.9, 0.05, 0.05],
                              [0.02, 0.8, 0.18],
                              [0.01, 0.1, 0.89]])
initial_ratings = np.array([0.7, 0.2, 0.1])  # Initial distribution (AAA, AA, A)

new_ratings = credit_rating_migration(transition_matrix, initial_ratings)
print(f"Rating distribution after one period: {new_ratings}")
