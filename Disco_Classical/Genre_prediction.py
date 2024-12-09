import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import librosa
from librosa import onset
import numpy as np
# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the K-Means model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load the cluster labels
with open('cluster_labels.pkl', 'rb') as f:
    cluster_labels = pickle.load(f)
# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(onset_env, max_size=1000)
    rhythmic_regularity = np.max(ac) / np.sum(ac)

    # Combine all features into a single feature vector
    features = np.hstack([
        chroma,
        tempo,
        spectral_centroid,
        zcr,
        mfcc_mean,
        rhythmic_regularity
    ])

    return features

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
    return cluster_labels[cluster]

# Define the directory and file naming format
directory = "C:/Users/User/OneDrive - American University of Beirut/Desktop/E3/EECE 490/MLproj/Data/Disco/"
file_format = "disco.{:05d}.wav"  # Formats numbers as 00001, 00002, etc.

# Loop through the range and generate file paths
for i in range(1, 2):
    file_path = directory + file_format.format(i)
    print(f"This song is: {predict_genre(file_path)}")
