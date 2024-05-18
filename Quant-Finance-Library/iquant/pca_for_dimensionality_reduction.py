import numpy as np

from sklearn.decomposition import PCA

def pca_for_dimensionality_reduction(data):
  """
  Performs PCA for dimensionality reduction on a dataset.

  Args:
      data: A NumPy array representing the data (rows: observations, columns: features).

  Returns:
      A NumPy array representing the data projected onto the principal components.
  """
  pca = PCA(n_components=2)  # Reduce to 2 principal components
  pca_data = pca.fit_transform(data)
  return pca_data

# Example usage (replace with actual data)
data = np.random.rand(100, 10)  # Sample data with 100 observations and 10 features

reduced_data = pca_for_dimensionality_reduction(data)
print(f"Shape of data after PCA: {reduced_data.shape}")
