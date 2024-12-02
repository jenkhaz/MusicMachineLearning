import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from preprocess import send_features


def k_means_clustering(data, n_clusters):
    """
    Perform k-means clustering on preprocessed audio features.

    Parameters:
        data (np.ndarray): Preprocessed and standardized features.
        n_clusters (int): Number of clusters for k-means.

    Returns:
        dict: Clustering results containing cluster labels, centroids, and the k-means model.
    """
    # Step 1: Fit KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)  # Predict clusters

    # Step 2: Extract centroids and return results
    return {
        "cluster_labels": cluster_labels,  # Cluster assignments
        "centroids": kmeans.cluster_centers_,  # Cluster centroids
        "kmeans_model": kmeans  # Trained k-means model
    }

def evaluate_clustering(data, labels):
    """
    Evaluate clustering performance using Silhouette Score.

    Parameters:
        data (np.ndarray): Preprocessed and standardized features.
        labels (np.ndarray): Cluster labels assigned by the k-means algorithm.

    Returns:
        float: Silhouette Score for clustering quality.
    """
    return silhouette_score(data, labels)

def visualize_clusters(data, labels, n_clusters):
    """
    Visualize clusters in 2D space using PCA.

    Parameters:
        data (np.ndarray): Preprocessed and standardized features.
        labels (np.ndarray): Cluster labels assigned by the k-means algorithm.
        n_clusters (int): Number of clusters for visualization.

    Returns:
        None: Displays a scatter plot of the clusters.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)  # Reduce to 2D for visualization

    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=labels,
        cmap='viridis',
        s=50
    )
    plt.title(f'K-Means Clustering Visualization ({n_clusters} Clusters)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.grid()
    plt.show()

# Algorithm Workflow
# Step 2: Preprocess the Dataset
preprocessed_features = send_features()
# Step 3: Perform K-Means Clustering
n_clusters = 10  # Define the number of clusters
clustering_results = k_means_clustering(preprocessed_features, n_clusters)

# Step 4: Evaluate Clustering Quality
silhouette = evaluate_clustering(preprocessed_features, clustering_results['cluster_labels'])
print(f"Silhouette Score for {n_clusters} clusters: {silhouette}")

# Step 5: Visualize Clusters
visualize_clusters(preprocessed_features, clustering_results['cluster_labels'], n_clusters)
