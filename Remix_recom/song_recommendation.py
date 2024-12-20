import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Function to load artifacts
def load_artifacts():
    # Load the dataset
    data = pd.read_csv(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\audio_features_genres_with_segments.csv")
    print(f"Dataset loaded successfully. Number of rows: {len(data)}")

    # Load the trained Random Forest model, scaler, and label encoder
    rf_classifier = joblib.load(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\randomforest_2genres\rf_genre_classifier.pkl")
    scaler = joblib.load(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\randomforest_2genres\scaler.pkl")
    label_encoder = joblib.load(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\randomforest_2genres\label_encoder.pkl")
    print("Random Forest model, scaler, and label encoder loaded successfully.")

    return data, scaler, rf_classifier, label_encoder

# Function to extract features from an audio file
def extract_features(file_path):
    print(f"Extracting features from file: {file_path}")
    y, sr = librosa.load(file_path, sr=None)

    # Extract features
    features = {
        'Chroma': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
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
    print(f"Extracted features: {features}")
    return features

# Function to recommend songs for remixing
def recommend_songs(input_song_features, data, label_encoder, scaler, rf_classifier):
    feature_columns = [
        'Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate',
        'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5',
        'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
        'MFCC_11', 'MFCC_12', 'MFCC_13', 'Rhythmic_Regularity'
    ]

    # Ensure the input features match the expected features
    missing_features = set(feature_columns) - set(input_song_features.keys())
    print(f"Expected features: {feature_columns}")
    print(f"Input features: {list(input_song_features.keys())}")
    print(f"Missing features: {missing_features}")

    for feature in missing_features:
        input_song_features[feature] = 0

    # Add an extra placeholder feature to align with scaler
    input_features_df = pd.DataFrame([input_song_features], columns=feature_columns)
    if input_features_df.shape[1] < scaler.n_features_in_:
        input_features_df['Extra_Feature_18'] = 0  # Add missing extra feature

    print(f"Input features dataframe:\n{input_features_df}")
    input_scaled = scaler.transform(input_features_df)
    print(f"Scaled input features: {input_scaled}")

    # Predict the genre
    input_genre_index = rf_classifier.predict(input_scaled)[0]
    input_genre = label_encoder.inverse_transform([input_genre_index])[0]
    print(f"Predicted input genre: {input_genre}")

    # Determine the target genre
    target_genre = 'disco' if input_genre == 'classical' else 'classical'
    print(f"Target genre: {target_genre}")

    # Filter songs by target genre
    target_songs = data[data['Label'] == target_genre]
    print(f"Filtered {len(target_songs)} songs from target genre '{target_genre}'.")

    # Add extra placeholder feature to target_features
    target_features = target_songs[feature_columns].copy()
    if target_features.shape[1] < scaler.n_features_in_:
        target_features['Extra_Feature_18'] = 0

    target_scaled = scaler.transform(target_features)
    similarities = cosine_similarity(input_scaled, target_scaled).flatten()

    # Add similarity scores and sort
    target_songs = target_songs.copy()
    target_songs['Similarity'] = similarities
    recommendations = target_songs.sort_values(by='Similarity', ascending=False).head(5)
    print(f"Generated {len(recommendations)} recommendations.")

    return input_genre, target_genre, recommendations[['Label', 'Segment_File_Name', 'Similarity']]

# Main function
def main():
    # Load artifacts
    data, scaler, rf_classifier, label_encoder = load_artifacts()

    # Define features and process data
    feature_columns = [
        'Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate',
        'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5',
        'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
        'MFCC_11', 'MFCC_12', 'MFCC_13', 'Rhythmic_Regularity'
    ]
    data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors='coerce')
    data[feature_columns] = data[feature_columns].fillna(data[feature_columns].mean())

    # Add extra placeholder feature to the dataset
    if data.shape[1] < scaler.n_features_in_:
        data['Extra_Feature_18'] = 0

    # Extract features
    audio_file = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\Remix_recom\Anja Lechner, François Couturier - Vague  E la nave va_segment_90_120.wav"
    input_song_features = extract_features(audio_file)

    # Get recommendations
    input_genre, target_genre, recommendations = recommend_songs(input_song_features, data, label_encoder, scaler, rf_classifier)

    # Display results
    print(f"\nInput song genre: {input_genre}")
    print(f"Recommended songs from the {target_genre} genre:")
    print(recommendations.to_string(index=False))

    def plot_similarity_scores(recommendations):
    # Rename the x-axis labels to "Recommended Song 1", "Recommended Song 2", etc.
        recommendations = recommendations.copy()  # Avoid modifying the original DataFrame
        recommendations['Song Label'] = [f"Recommended Song {i+1}" for i in range(len(recommendations))]
        
        # Define custom colors
        colors = ['#1d434e', '#4eb6b0', '#830131', '#a96d83']
        bar_colors = colors * (len(recommendations) // len(colors)) + colors[:len(recommendations) % len(colors)]
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=recommendations, x='Song Label', y='Similarity', palette=bar_colors)
        plt.title('Similarity Scores for Recommended Songs')
        plt.xlabel('Recommended Songs')
        plt.ylabel('Similarity Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot as a file
        plt.savefig('recommended_songs_similarity_plot.png', dpi=300, bbox_inches='tight')
        plt.show()


        
    plot_similarity_scores(recommendations)

if __name__ == "__main__":
    main()
