import librosa # type: ignore
import numpy as np # type: ignore
import pickle
import pandas as pd

# Load the scaler, k-means model, and cluster labels
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('cluster_labels.pkl', 'rb') as f:
    cluster_labels = pickle.load(f)

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(onset_env, max_size=1000)
    rhythmic_regularity = np.max(ac) / np.sum(ac)
    
    features = np.hstack([
        chroma,
        tempo,
        spectral_centroid,
        zcr,
        mfcc,
        rhythmic_regularity
    ])
    return features

# Genre prediction

def predict_genre(song_path):
    # Extract features from the audio file
    features = extract_features(song_path)
    
    # Define the feature names (must match training dataset)
    feature_names = ['Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate', 
                     'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 
                     'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 
                     'MFCC_13', 'Rhythmic_Regularity']
    
    # Convert features to a DataFrame
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale the features using the pre-trained scaler
    features_scaled = scaler.transform(features_df)
    
    # Predict the cluster using the pre-trained KMeans model
    cluster = kmeans.predict(features_scaled)[0]
    
    # Map the cluster to the corresponding genre label
    return (cluster_labels[cluster],features_scaled)
def transform_to_target_cluster(features_scaled, controllable_indices, source_cluster, target_cluster, alpha=0.1, steps=10):
    """
    Transform features of a song to align more closely with the target cluster, returning the musician-readable format.
    """
    # Get centroids for source and target clusters
    source_centroid = kmeans.cluster_centers_[source_cluster]
    target_centroid = kmeans.cluster_centers_[target_cluster]

    # Initialize feature vector
    current_features = features_scaled[0].copy()

    # Keep a copy of the original feature vector
    original_features = current_features.copy()

    for step in range(steps):
        # Update only controllable indices
        for idx in controllable_indices:
            # Move controllable features toward the target cluster centroid
            current_features[idx] += alpha * (target_centroid[idx] - current_features[idx])

        # Compute the current distance to the target centroid (controllable features only)
        controllable_distance = np.linalg.norm(
            current_features[controllable_indices] - target_centroid[controllable_indices]
        )
        print(f"Step {step}, Distance to Target Centroid (Controllable Features): {controllable_distance}")

        # Stop if the adjusted features belong to the target cluster
        if kmeans.predict([current_features])[0] == target_cluster:
            print(f"Reached target cluster at step {step}")
            break

    # Combine updated controllable features with fixed original features for non-controllable indices
    for idx in range(len(current_features)):
        if idx not in controllable_indices:
            current_features[idx] = original_features[idx]

    # Inverse transform the final features to their original scale
    transformed_features_original_scale = scaler.inverse_transform([current_features])
    
    # Return only the controllable features in their original scale
    transformed_controllable_features = transformed_features_original_scale[0][controllable_indices]
    return transformed_controllable_features, transformed_features_original_scale



# Extract features
song_path = "C:/Users/User/OneDrive - American University of Beirut/Desktop/E3/EECE 490/MLproj/Data/Classical/01.wav"
features = extract_features(song_path)
_, features_scaled = predict_genre(song_path)

# Define controllable indices (e.g., tempo, rhythmic regularity, MFCCs)
controllable_indices = [1, -1]  # Tempo and Rhythmic Regularity (adjust based on your feature set)

# Identify source and target clusters
source_cluster = 0  # Classical
target_cluster = 1  # Disco
# Transform features
transformed_controllable_features, transformed_features_original_scale = transform_to_target_cluster(
    features_scaled, controllable_indices, source_cluster, target_cluster
)

# Print the results
print("Original Features (Scaled):", features_scaled)
print("Transformed Controllable Features (Musician Format):", transformed_controllable_features)
print("Entire Transformed Features (Original Scale):", transformed_features_original_scale)

# Check the cluster of the transformed features
new_cluster = kmeans.predict([transformed_features_original_scale[0]])[0]
print(f"New cluster: {new_cluster}, which is the {cluster_labels[new_cluster]} cluster")


"""
transformed_features = transform_to_target_cluster(features_scaled, controllable_indices, source_cluster, target_cluster)

print("Transformed Features (Original Scale):", transformed_features)
#else:
   # print("We can only recommend changes applicable to classical songs, sorry :(")
   # Ensure adjusted_features is reshaped into 2D array
#adjusted_features = adjusted_features.reshape(1, -1)
print("orginial",features_scaled)
new_cluster = kmeans.predict(transformed_features)[0]
print (f"New cluster: {new_cluster}, which is the {cluster_labels[new_cluster]} cluster")
#old_cluster=kmeans.predict(features)[0]
#print (f"New cluster: {old_cluster}, which is the {cluster_labels[old_cluster]} cluster")
"""