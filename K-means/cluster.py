from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from preprocess import send_features

def cluster_audio_features(data, n_clusters=3):
    """
    Perform k-means clustering on preprocessed audio features.

    Parameters:
        data (np.ndarray): Preprocessed and standardized audio features.
        n_clusters (int): Number of clusters for k-means.

    Returns:
        dict: Clustering results containing cluster labels and reduced dimensions for visualization.
    """
    # Step 1: Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    
    # Step 2: Dimensionality Reduction for Visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.grid()
    plt.show()
    
    return {
        "cluster_labels": cluster_labels,
        "reduced_data": reduced_data,
        "kmeans_model": kmeans
    }

preprocessed_features = send_features()
# Perform clustering
clustering_results = cluster_audio_features(preprocessed_features, n_clusters=3)
