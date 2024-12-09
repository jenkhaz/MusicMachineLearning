
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

def main():
    # Load the dataset
    data = pd.read_csv('audio_features(disco_classical).csv')
  

    # Remove brackets from columns and ensure numeric conversion
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = data[column].astype(str).str.strip("[]").astype(float, errors='ignore')

    # Separate features and labels
    X = data.drop(columns=['Label'])
    y_true = data['Label']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Map clusters to predominant genres
    cluster_labels = {}
    for cluster in range(kmeans.n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        predominant_genre = cluster_data['Label'].mode()[0]
        cluster_labels[cluster] = predominant_genre

    data['Predicted_Label'] = data['Cluster'].map(cluster_labels)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data['PCA1'] = X_pca[:, 0]
    data['PCA2'] = X_pca[:, 1]

    # Visualize clusters
    plt.figure(figsize=(10, 8))
    for cluster, color in zip(data['Cluster'].unique(), ['blue', 'red']):
        subset = data[data['Cluster'] == cluster]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=f"Cluster {cluster}", alpha=0.6, c=color)
    plt.title('PCA Visualization of Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.savefig('cluster_visualization.png')
    plt.show()
    print("Cluster-based plot saved as 'cluster_visualization.png'")

    # Visualize original labels
    plt.figure(figsize=(10, 8))
    for label, color in zip(y_true.unique(), ['green', 'orange']):
        subset = data[data['Label'] == label]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=f"{label} (Original)", alpha=0.6, c=color)
    plt.title('PCA Visualization by Original Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.savefig('original_label_visualization.png')
    plt.show()
    print("Original label plot saved as 'original_label_visualization.png'")

    # Evaluate clustering accuracy
    correct = sum(data['Predicted_Label'] == y_true)
    accuracy = correct / len(data) * 100
    print(f"\nClustering accuracy: {accuracy:.2f}%")

    # Save the updated dataset
    data.to_csv('audio_features_with_clusters_and_pca(classical_disco).csv', index=False)
    print("Clustering and PCA completed. Results saved to 'audio_features_with_clusters_and_pca.csv'.")

    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save the K-Means model
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    # Save the cluster labels
    with open('cluster_labels.pkl', 'wb') as f:
        pickle.dump(cluster_labels, f)

if __name__ == "__main__":
    main()
