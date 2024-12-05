import os
import librosa
from librosa import onset
import numpy as np
import pandas as pd

# Dynamically determine the base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
rock_path = os.path.join(base_path, r'C:\Users\Lenovo\Desktop\data\rock-wav')

# Function to extract features
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
    return [chroma, tempo, spectral_centroid, zcr, *mfcc, rhythmic_regularity]

# Process the rock dataset
def process_dataset(dataset_path, label):
    features = []
    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)
        try:
            data = extract_features(file_path)
            data.append(label)  # Add genre label
            features.append(data)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    return features

# Extract features for rock genre
rock_features = process_dataset(rock_path, 'rock')

# Define columns for the DataFrame
columns = ['Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate', 
           'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 
           'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 
           'MFCC_13', 'Rhythmic_Regularity', 'Label']

# Read the existing CSV file
csv_file = r'C:\Users\Lenovo\Desktop\MusicMachineLearning\output_3genres\audio_features(disco_classical_techno).csv'
df_existing = pd.read_csv(csv_file)

# Create a DataFrame for rock features
df_rock = pd.DataFrame(rock_features, columns=columns)

# Append rock features to the existing DataFrame
df_updated = pd.concat([df_existing, df_rock], ignore_index=True)

# Save the updated DataFrame to CSV
df_updated.to_csv(csv_file, index=False)

print(f"Rock features added and saved to '{csv_file}'")
