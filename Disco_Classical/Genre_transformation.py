import librosa
import numpy as np
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
    current_cluster = kmeans.predict(features_scaled)[0]
    target_cluster = [key for key, value in cluster_labels.items() if value.lower() == 'disco'][0]
    
    centroids = kmeans.cluster_centers_
    current_centroid = centroids[current_cluster]
    target_centroid = centroids[target_cluster]
    
    chroma_index = 0  # Assuming chroma is the first feature
    chroma_feature = features_scaled[0][chroma_index]
    
    # Exclude chroma from transformation
    non_harmonic_features = np.delete(features_scaled[0], chroma_index)
    current_non_harmonic = np.delete(current_centroid, chroma_index)
    target_non_harmonic = np.delete(target_centroid, chroma_index)
    
    # Compute the transformation vector for non-harmonic features
    transformation_vector = target_non_harmonic - current_non_harmonic
    scale_factor = 1.0
    new_non_harmonic_features = non_harmonic_features + scale_factor * transformation_vector
    
    # Reconstruct features, keeping chroma unchanged
    new_features = np.insert(new_non_harmonic_features, chroma_index, chroma_feature)
    
    # Restore original chroma after inverse scaling
    new_song_features = scaler.inverse_transform([new_features])
    new_song_features[0][chroma_index] = scaler.inverse_transform(features_scaled)[0][chroma_index]
    
    return transformation_vector, new_song_features


song_path = input("Please input the song's path: ")

# Extract features
features = extract_features(song_path)

# Predict the genre using the file path, not the features
Current_genre, features = predict_genre(song_path)
print (Current_genre)
if Current_genre == 'classical':
    transformation_vector, adjusted_features = compute_transformation_vector(features)
    print("Original Features:", features)
    print("Transformation Vector:", transformation_vector)
    print("Adjusted Features (in original scale):", adjusted_features)
   
else:
    print("We can only recommend changes applicable to classical songs, sorry :(")
