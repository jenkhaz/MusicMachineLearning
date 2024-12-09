from autoencoder import prepare_dataset, create_autoencoder, extract_features
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data_dir = 'C:\\Users\\user\\musicml_490\\MusicMachineLearning\\autoencoder_kmeans\\songs_gtzan'
features = prepare_dataset(data_dir)
input_shape = features.shape[1:]

autoencoder, encoder = create_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(features, features, epochs=50, batch_size=16, shuffle=True)

# Step 4: Extract Feature Vectors
feature_vectors = encoder.predict(features)
#need to flatten the array since it's 3D into 2D to feed into k-means model
feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)


# Step 5: K-Means Clustering
kmeans = KMeans(n_clusters=7)  # Testing over 7 genres
clusters = kmeans.fit_predict(feature_vectors)

print("Cluster Assignments:", clusters)


# Assume `feature_vectors` contains your feature vectors from the encoder
# and `clusters` contains the k-means cluster assignments.

# Step 1: Dimensionality Reduction (choose PCA or t-SNE)
# Option 1: Using PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_vectors)

# Option 2: Using t-SNE (uncomment to use t-SNE instead)
#t-SNE is a non-linear method that's better than PCA but it's slower
# tsne = TSNE(n_components=2, random_state=42)
# reduced_features = tsne.fit_transform(feature_vectors)

# Step 2: Plot the Clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('Visualization of Clusters')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()