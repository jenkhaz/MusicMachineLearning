import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the dataset and clean brackets
data = pd.read_csv('audio_features.csv')

# Remove brackets from columns and ensure numeric conversion
for column in data.columns:
    if data[column].dtype == 'object':  # Check if column contains strings
        data[column] = data[column].astype(str).str.strip("[]").astype(float, errors='ignore')

# Separate features and labels
X = data.drop(columns=['Label'])  # Features only
y_true = data['Label']  # True labels for evaluation

# Step 2: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply k-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Add predictions to the dataset
data['Cluster'] = kmeans.labels_

# Step 4: Check Label Distribution in Each Cluster
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    label_counts = cluster_data['Label'].value_counts()
    print(f"\nCluster {cluster}:")
    print(label_counts)

# Map cluster numbers to genres based on majority voting
mapping = {}
for cluster in data['Cluster'].unique():
    genre = data[data['Cluster'] == cluster]['Label'].mode()[0]
    mapping[cluster] = genre

data['Predicted_Label'] = data['Cluster'].map(mapping)

# Step 5: Apply PCA for visualization
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

# Add PCA results to the dataset
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]
plt.figure(figsize=(10, 8))
for cluster, color in zip(data['Cluster'].unique(), ['blue', 'red']):
    subset = data[data['Cluster'] == cluster]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=f"Cluster {cluster}", alpha=0.6, c=color)

plt.title('PCA Visualization of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()

# Save and show the plot
plt.savefig('cluster_visualization.png')  # Save the plot
plt.show(block=True)  # Display the plot
print("Cluster-based plot saved as 'cluster_visualization.png'")

# Step 7: Visualize PCA Points by Original Labels
plt.figure(figsize=(10, 8))
for label, color in zip(y_true.unique(), ['green', 'orange']):  # Assign colors to original labels
    subset = data[data['Label'] == label]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=f"{label} (Original)", alpha=0.6, c=color)

plt.title('PCA Visualization by Original Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()

# Save and show the plot
plt.savefig('original_label_visualization.png')  # Save the plot
plt.show(block=True)  # Display the plot
print("Original label plot saved as 'original_label_visualization.png'")

# Step 8: Evaluate clustering accuracy
correct = sum(data['Predicted_Label'] == y_true)
accuracy = correct / len(data) * 100
print(f"\nClustering accuracy: {accuracy:.2f}%")

# Step 9: Save the updated dataset with clusters
data.to_csv('audio_features_with_clusters_and_pca.csv', index=False)
print("Clustering and PCA completed. Results saved to 'audio_features_with_clusters_and_pca.csv'.")
