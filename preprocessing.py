import os
import librosa
from librosa import onset
import numpy as np
import pandas as pd

# Dynamically determine the base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
classical_path = os.path.join(base_path, 'C:/Users/User/OneDrive - American University of Beirut/Desktop/E3/EECE 490/MLproj/Data/Classical')
techno_path= os.path.join(base_path, 'C:/Users/User/OneDrive - American University of Beirut/Desktop/E3/EECE 490/MLproj/Data/Tech')

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
    return [chroma, tempo, spectral_centroid, zcr, *mfcc,rhythmic_regularity]


# Process the datasets
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

# Extract features for both genres
classical_features = process_dataset(classical_path, 'classical')
techno_features= process_dataset(techno_path, 'techno')

# Define columns for the DataFrame
columns = ['Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate', 
           'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 
           'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 
           'MFCC_13', 'Rhythmic_Regularity', 'Label']

# Create a DataFrame
df = pd.DataFrame(classical_features +techno_features, columns=columns)

# Save to CSV
df.to_csv('audio_features(classical_techno).csv', index=False)

print("Feature extraction completed and saved to 'audio_features(classical_techno).csv'")
