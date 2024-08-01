import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = '/mnt/data/sd254_users.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display the summary of the dataset
print("\nSummary of the dataset:")
data.info()

# Display summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(data.describe())

# Preprocess the data
# Identify numerical and categorical columns
numerical_cols = ['age', 'income']
categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'user_id']

# Preprocessing pipeline for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Apply preprocessing to the data
X = data.drop(columns=['user_id', 'spending_score'])
X_preprocessed = preprocessor.fit_transform(X)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocessed)

# Find the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.grid(True)
plt.show()

# Choose the optimal number of clusters (e.g., k=4 based on the elbow plot)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X_pca)

# Evaluate the clustering using the silhouette score
sil_score = silhouette_score(X_pca, data['cluster'])
print(f'Silhouette Score: {sil_score:.2f}')

# Visualize the clusters in the PCA-reduced feature space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segments Based on PCA-Reduced Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Analyze the clusters
print("\nCluster Analysis:")
for cluster in range(optimal_clusters):
    print(f"\nCluster {cluster}:")
    cluster_data = data[data['cluster'] == cluster]
    print(cluster_data.describe())

# Optionally, save the clustered data for further analysis
output_file_path = '/mnt/data/clustered_sd254_users.csv'
data.to_csv(output_file_path, index=False)
print(f"\nClustered data saved to {output_file_path}")
