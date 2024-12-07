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
# Compute transformation vector
def compute_transformation_vector(features_scaled):
    # Identify the current cluster
    current_cluster = kmeans.predict(features_scaled)[0]
    
    # Determine the target cluster (assuming binary clustering)
    target_cluster = 1 if current_cluster == 0 else 0
    
    # Retrieve cluster centroids
    centroids = kmeans.cluster_centers_
    current_centroid = centroids[current_cluster]
    target_centroid = centroids[target_cluster]
    
    # Compute the hyperplane parameters
    w = target_centroid - current_centroid
    b = (np.dot(target_centroid, target_centroid) - np.dot(current_centroid, current_centroid)) / 2.0
    
    # Preserve the chroma feature (assumed to be the first feature)
    chroma_index = 0
    chroma_feature = features_scaled[0][chroma_index]
    
    # Compute the offset b'
    b_prime = b - w[chroma_index] * chroma_feature
    
    # Initialize new features with original values
    new_features = features_scaled[0].copy()
    
    # Adjust non-chroma features
    non_chroma_indices = [i for i in range(len(new_features)) if i != chroma_index]
    w_non_chroma = w[non_chroma_indices]
    original_non_chroma = new_features[non_chroma_indices]
    
    # Solve for the adjustment
    adjustment = (b_prime - np.dot(w_non_chroma, original_non_chroma)) / np.sum(w_non_chroma)
    
    # Apply the adjustment proportionally
    new_non_chroma_features = original_non_chroma + adjustment * (w_non_chroma / np.linalg.norm(w_non_chroma))
    
    # Update the new features
    new_features[non_chroma_indices] = new_non_chroma_features
    
    # Restore chroma feature explicitly before scaling back
    new_features[chroma_index] = chroma_feature
    
    # Inverse transform to original scale
    new_song_features = scaler.inverse_transform([new_features])
    
    # Restore the chroma feature in the original scale explicitly
    original_features_in_original_scale = scaler.inverse_transform(features_scaled)
    new_song_features[0][chroma_index] = original_features_in_original_scale[0][chroma_index]
    
    return new_song_features

# Extract features
song_path = "C:/Users/User/OneDrive - American University of Beirut/Desktop/E3/EECE 490/MLproj/Data/Classical/01.wav"
features = extract_features(song_path)
# Predict the genre using the file path, not the features
Current_genre, features = predict_genre(song_path)
print (Current_genre)
if Current_genre == 'classical':
    adjusted_features = compute_transformation_vector(features)
    print("Original Features:", features)
    
    print("Adjusted Features (in original scale):", adjusted_features)
   
else:
    print("We can only recommend changes applicable to classical songs, sorry :(")
