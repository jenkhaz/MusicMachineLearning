import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import librosa  # For audio feature extraction
import numpy as np

# Function to load artifacts: dataset, scaler, KMeans model, and cluster labels
def load_artifacts():
    # Load the preprocessed data
    data = pd.read_csv(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\audio_features_genres_with_segments.csv")
    print("Dataset loaded successfully. Number of rows:", len(data))

    # Load the scaler
    with open(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\scaler_2_clusters.pkl", 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")

    # Load the K-Means model
    with open(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\kmeans_model_2_clusters.pkl", 'rb') as f:
        kmeans = pickle.load(f)
    print("KMeans model loaded successfully.")

    # Load the cluster labels
    with open(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\cluster_labels_2_clusters.pkl", 'rb') as f:
        cluster_labels = pickle.load(f)
    print("Cluster labels loaded successfully:", cluster_labels)

    return data, scaler, kmeans, cluster_labels

# Function to extract features from an audio file
def extract_features(file_path):
    print(f"Extracting features from file: {file_path}")
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    features = {
        'Chroma': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        # Use librosa.beat.tempo for backward compatibility
        'Tempo': librosa.beat.tempo(y=y, sr=sr)[0],
        'Spectral_Centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'Zero_Crossing_Rate': np.mean(librosa.feature.zero_crossing_rate(y=y)),
    }
    
    # Add MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'MFCC_{i+1}'] = np.mean(mfccs[i])
    
    # Add additional rhythmic features
    features['Rhythmic_Regularity'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    print("Features extracted successfully.")
    return features

# Function to recommend songs for remixing
def recommend_songs(input_song_features, data, cluster_labels, scaler, kmeans, known_genre=None):
    feature_columns = [
        'Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate',
        'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5',
        'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
        'MFCC_11', 'MFCC_12', 'MFCC_13', 'Rhythmic_Regularity'
    ]
    input_features_df = pd.DataFrame([input_song_features], columns=feature_columns)
    input_scaled = scaler.transform(input_features_df)
    print(f"Input scaled features: {input_scaled}")

    # Predict the cluster and get its genre
    input_cluster = kmeans.predict(input_scaled)[0]
    input_genre = cluster_labels[input_cluster]
    print(f"Predicted input cluster: {input_cluster}, genre: {input_genre}")

    # Override genre if provided
    if known_genre:
        input_genre = known_genre
        input_cluster = [k for k, v in cluster_labels.items() if v == known_genre][0]
        print(f"Overriding input genre to: {known_genre}")

    # Opposite cluster selection
    target_genre = 'disco' if input_genre == 'classical' else 'classical'
    target_cluster = [k for k, v in cluster_labels.items() if v == target_genre][0]
    print(f"Target cluster: {target_cluster}, target genre: {target_genre}")

    # Filter songs by target cluster and genre
    target_songs = data[(data['Cluster'] == target_cluster) & (data['Label'] == target_genre)]
    print(f"Filtered {len(target_songs)} songs from target genre '{target_genre}'.")

    if target_songs.empty:
        print("No songs available in the target cluster for recommendations.")
        return input_genre, target_genre, pd.DataFrame()

    target_features = target_songs[feature_columns]
    print(f"Target features shape: {target_features.shape}")
    similarities = cosine_similarity(input_scaled, scaler.transform(target_features)).flatten()
    print(f"Similarities: {similarities}")

    # Add similarity scores
    target_songs = target_songs.copy()
    target_songs['Similarity'] = similarities
    recommendations = target_songs.sort_values(by='Similarity', ascending=False).head(5)
    print(f"Generated {len(recommendations)} recommendations.")

    return input_genre, target_genre, recommendations[['Label', 'Segment_File_Name', 'Similarity']]

# Main function
# Main function
def main():
    # Load artifacts and dataset
    data, scaler, kmeans, cluster_labels = load_artifacts()

    # Define feature columns
    feature_columns = [
        'Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate',
        'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5',
        'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
        'MFCC_11', 'MFCC_12', 'MFCC_13', 'Rhythmic_Regularity'
    ]

    # Clean and convert feature columns
    for column in feature_columns:
        data[column] = data[column].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)

    # Handle non-numeric values
    data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors='coerce')
    data[feature_columns] = data[feature_columns].fillna(data[feature_columns].mean())

    # Add the Cluster column
    data['Cluster'] = kmeans.predict(scaler.transform(data[feature_columns]))
    print("Cluster column added to dataset.")

    # Set the path to the audio file
    audio_file = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\Remix_recom\Anja Lechner, François Couturier - Vague  E la nave va_segment_90_120.wav"

    # Extract features from the input song
    input_song_features = extract_features(audio_file)
    print("\nExtracted Features:")
    for feature, value in input_song_features.items():
        print(f"{feature}: {value}")

    # Known genre for debugging (set to 'classical' or 'disco' if applicable)
    known_genre = None  # Change to 'classical' or 'disco' for testing

    # Get recommendations
    input_genre, target_genre, recommendations = recommend_songs(input_song_features, data, cluster_labels, scaler, kmeans, known_genre)

    # Print cluster and label distributions
    print("Cluster distribution:\n", data['Cluster'].value_counts())
    print("Label distribution:\n", data['Label'].value_counts())

    # Print extracted input features
    print(f"Extracted input features: {input_song_features}")

    # Print dataset feature ranges
    print(f"Dataset feature ranges: {data[feature_columns].min().to_dict()} to {data[feature_columns].max().to_dict()}")

    # Display results
    print(f"\nInput song genre: {input_genre}")
    print(f"Recommended songs from the {target_genre} genre:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
