import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf  # For saving audio files

# Dynamically determine the base path
base_path = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths for each genre
genres = {
    'classical': os.path.join(base_path, r"C:\Users\Lenovo\Desktop\data\Classical"),
    'techno': os.path.join(base_path, r"C:\Users\Lenovo\Desktop\data\Techno"),
    'disco': os.path.join(base_path, r"C:\Users\Lenovo\Desktop\data\Disco"),
    'rock': os.path.join(base_path, r"C:\Users\Lenovo\Desktop\data\rock")
}

# Directory to save segmented audio files
output_dir = os.path.join(base_path, 'Segmented_Audio')
os.makedirs(output_dir, exist_ok=True)

# Function to extract features
def extract_features(y, sr):
    if np.sum(np.abs(y)) < 1e-3:  # Adjust the threshold as needed
        raise ValueError("Silent or near-silent audio segment detected.")

    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(onset_env, max_size=1000)
    rhythmic_regularity = np.max(ac) / np.sum(ac)
    return [chroma, tempo, spectral_centroid, zcr, *mfcc, rhythmic_regularity]

# Process each song by splitting into 30-second segments
def process_song(file_path, label):
    y, sr = librosa.load(file_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    segment_features = []
    for start in range(0, int(duration), 30):
        end = start + 30
        segment = y[start * sr:end * sr] if end * sr <= len(y) else y[start * sr:]
        if len(segment) < sr:  # Skip very small segments
            continue
        features = extract_features(segment, sr)

        # Save the segmented audio
        file_name = os.path.basename(file_path)
        segment_file_name = f"{os.path.splitext(file_name)[0]}_segment_{start}_{end}.wav"
        genre_dir = os.path.join(output_dir, label)
        os.makedirs(genre_dir, exist_ok=True)
        segment_path = os.path.join(genre_dir, segment_file_name)
        sf.write(segment_path, segment, sr)

        # Add features along with the filename and genre
        features.append(segment_file_name)
        features.append(label)
        segment_features.append(features)
    return segment_features

# Process the datasets
def process_dataset(dataset_path, label):
    features = []
    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)
        try:
            features += process_song(file_path, label)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    return features

# Aggregate features for all genres
all_features = []
for genre, path in genres.items():
    all_features += process_dataset(path, genre)

# Define columns for the DataFrame
columns = ['Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate', 
           'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 
           'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 
           'MFCC_13', 'Rhythmic_Regularity', 'Segment_File_Name', 'Label']

# Create a DataFrame
df = pd.DataFrame(all_features, columns=columns)

# Save to CSV
df.to_csv('audio_features_genres_with_segments.csv', index=False)

print(f"Feature extraction completed. Features saved to 'audio_features_genres_with_segments.csv', and segmented audio files are saved in '{output_dir}'.")
