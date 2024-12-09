import numpy as np
import librosa
import os
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logging.basicConfig(level=logging.INFO)

# Step 1: Preprocess the audio
def extract_features(file_path, n_mfcc=13, max_length=130):
    try:
        audio, sr = librosa.load(file_path, duration=30)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] > max_length:
            mfccs = mfccs[:, :max_length]
        else:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        return mfccs.T
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Step 2: Build the Encoder-Decoder
def create_autoencoder(input_shape):
    print("Input Shape:", input_shape)  # Should be (num_samples, 130, 13)

    input_layer = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(32, activation='relu', name='encoder_output')(x)

    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(input_shape[-1], activation='sigmoid')(x)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)
    return autoencoder, encoder

# Step 3: Prepare Dataset and Train
def prepare_dataset(data_dir, n_mfcc=13):
    features = []
   # labels=[]
    for file in os.listdir(data_dir):
        if file.endswith('.mp3') or file.endswith('.wav'):
            logging.info(f"Processing file: {file}")
            file_path = os.path.join(data_dir, file)
            features.append(extract_features(file_path, n_mfcc))
    return np.array(features)

# Training Parameters
data_dir = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\audio_features_genres_with_segments.csv"
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