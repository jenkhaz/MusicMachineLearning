"""
This script applies K-Means clustering to a dataset of audio features from various music genres.
It then evaluates and visualizes the resulting clusters.

Steps:
1. Load and preprocess data (remove brackets, convert to numeric).
2. Extract features, normalize them with StandardScaler.
3. Perform K-Means clustering (n_clusters=4).
4. Assign each cluster to the predominant genre found within it.
5. Compute accuracy against original labels.
6. Reduce dimensions with PCA and visualize both clusters and original labels.
7. Save results, models, and plots.

Generates:
- cluster_visualization_4_clusters.png: PCA plot by assigned clusters.
- original_label_visualization_4_clusters.png: PCA plot by original labels.
- audio_features_with_clusters_and_pca_4_clusters.csv: Data with cluster assignments and PCA components.
- scaler.pkl: Fitted StandardScaler.
- kmeans_model_4_clusters.pkl: Trained K-Means model.
- cluster_labels_4_clusters.pkl: Mapping of cluster indices to genres.
"""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

def main():
    # Load the dataset
    data = pd.read_csv(r"audio_features_genres_with_segments.csv")

    # Remove brackets from columns and ensure numeric conversion
    for column in data.columns:
        if data[column].dtype == 'object':
            try:
                data[column] = data[column].astype(str).str.strip("[]").astype(float, errors='ignore')
            except ValueError:
                pass  # Skip non-numeric columns

    # Separate features and labels
    non_feature_columns = ['Label', 'Segment_File_Name']  # Non-numeric columns to exclude
    X = data.drop(columns=non_feature_columns, errors='ignore')
    y_true = data['Label']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering with 4 clusters (one for each genre)
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Map clusters to predominant genres
    cluster_labels = {}
    for cluster in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        predominant_genre = cluster_data['Label'].mode()[0]
        cluster_labels[cluster] = predominant_genre

    data['Predicted_Label'] = data['Cluster'].map(cluster_labels)

    # Calculate accuracy
    correct = sum(data['Predicted_Label'] == y_true)
    accuracy = correct / len(data) * 100
    print(f"\nClustering accuracy: {accuracy:.2f}%")

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data['PCA1'] = X_pca[:, 0]
    data['PCA2'] = X_pca[:, 1]

    # Visualize clusters
    plt.figure(figsize=(10, 8))
    colors = ['#1d434e', '#4eb6b0', '#830131', '#a96d83']
    for cluster, color in zip(data['Cluster'].unique(), colors):
        subset = data[data['Cluster'] == cluster]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=f"Cluster {cluster}", alpha=0.6, c=color)
    plt.title(f'PCA Visualization of Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.savefig('cluster_visualization_4_clusters.png')
    plt.show()
    print("Cluster-based plot saved as 'cluster_visualization_4_clusters.png'")

    # Visualize original labels
    plt.figure(figsize=(10, 8))
    label_colors = ['#1d434e', '#4eb6b0', '#830131', '#a96d83']
    for label, color in zip(y_true.unique(), label_colors):
        subset = data[data['Label'] == label]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=f"{label} (Original)", alpha=0.6, c=color)
    plt.title('PCA Visualization by Original Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.savefig('original_label_visualization_4_clusters.png')
    plt.show()
    print("Original label plot saved as 'original_label_visualization_4_clusters.png'")

    # Save the updated dataset
    data.to_csv('audio_features_with_clusters_and_pca_4_clusters.csv', index=False)
    print("Clustering and PCA completed. Results saved to 'audio_features_with_clusters_and_pca_4_clusters.csv'.")

    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save the K-Means model
    with open('kmeans_model_4_clusters.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    # Save the cluster labels
    with open('cluster_labels_4_clusters.pkl', 'wb') as f:
        pickle.dump(cluster_labels, f)

if __name__ == "__main__":
    main()
