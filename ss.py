# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the data
df = pd.read_csv(r'C:\Users\bk\Downloads\project-files-customer-segmentation-in-marketing-with-python\customer_segmentation_data.csv')

# Data Preprocessing
columns_to_drop = ['Row ID', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Customer ID']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)  # Drop unnecessary columns
df = df.dropna()  # Remove missing values
df = df.reset_index(drop=True)  # Reset index

# Scaling the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Hierarchical Clustering
# Perform linkage for hierarchical clustering
linkage_matrix = linkage(df_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Determine clusters by cutting the dendrogram
# Change the number (e.g., 2, 4, 5, 8) to set the number of clusters
n_clusters = 4
cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

# Add cluster labels to the original dataset
df['Cluster'] = cluster_labels

# Visualize Clusters (Optional PCA for visualization)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = cluster_labels

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title('Customer Segmentation (Hierarchical Clustering)')
plt.savefig('hierarchical_customer_segmentation.png')
plt.show()

# Evaluate Clustering Performance
silhouette_avg = silhouette_score(df_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(df_scaled, cluster_labels)

print(f'Number of Clusters: {n_clusters}')
print(f'Silhouette Score: {silhouette_avg:.4f}')
print(f'Davies-Bouldin Index: {davies_bouldin:.4f}')

# Save results to a CSV file
df.to_csv(r'C:\Users\bk\Downloads\hierarchical_clustering_results.csv', index=False)
print("Clustered dataset saved as 'hierarchical_clustering_results.csv'.")
