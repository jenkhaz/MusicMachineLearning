import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score

def main():
    # Load the dataset
    data = pd.read_csv(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\audio_features_genres_with_segments.csv")

    # Filter the dataset to include only Disco and Classical genres
    data = data[data['Label'].isin(['disco', 'classical'])]

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

    # Apply K-Means clustering with 2 clusters (one for each genre)
    kmeans = KMeans(n_clusters=2, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Map clusters to predominant genres
    cluster_labels = {}
    for cluster in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        predominant_genre = cluster_data['Label'].mode()[0]
        cluster_labels[cluster] = predominant_genre

    data['Predicted_Label'] = data['Cluster'].map(cluster_labels)

    # Evaluate clustering accuracy
    y_pred = data['Predicted_Label']
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\nClustering accuracy: {accuracy:.2f}%")

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data['PCA1'] = X_pca[:, 0]
    data['PCA2'] = X_pca[:, 1]

    # Visualize clusters
    plt.figure(figsize=(10, 8))
    colors = ['#1d434e', '#4eb6b0']
    for cluster, color in zip(data['Cluster'].unique(), colors):
        subset = data[data['Cluster'] == cluster]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=f"Cluster {cluster}", alpha=0.6, c=color)
    plt.title('PCA Visualization of Clusters (Disco and Classical)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.savefig('cluster_visualization_2_clusters.png')
    plt.show()
    print("Cluster-based plot saved as 'cluster_visualization_2_clusters.png'")

    # Visualize original labels
    plt.figure(figsize=(10, 8))
    label_colors = ['#4eb6b0', '#970c10']
    for label, color in zip(y_true.unique(), label_colors):
        subset = data[data['Label'] == label]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=f"{label} (Original)", alpha=0.6, c=color)
    plt.title('PCA Visualization by Original Labels (Disco and Classical)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.savefig('original_label_visualization_2_clusters.png')
    plt.show()
    print("Original label plot saved as 'original_label_visualization_2_clusters.png'")

    # Save the updated dataset
    data.to_csv('audio_features_with_clusters_and_pca_2_clusters.csv', index=False)
    print("Clustering and PCA completed. Results saved to 'audio_features_with_clusters_and_pca_2_clusters.csv'.")

    # Save the scaler
    with open('scaler_2_clusters.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save the K-Means model
    with open('kmeans_model_2_clusters.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    # Save the cluster labels
    with open('cluster_labels_2_clusters.pkl', 'wb') as f:
        pickle.dump(cluster_labels, f)

if __name__ == "__main__":
    main()
