import numpy as np
import joblib
import librosa

def extract_features(file_path):
    """
    Extracts audio features from the provided file and ensures all expected features are present.

    Parameters:
        file_path (str): Path to the audio file (.wav).

    Returns:
        list: A list of extracted features in the correct order.
    """
    print(f"Extracting features from file: {file_path}")
    y, sr = librosa.load(file_path, sr=None)

    # Initialize a dictionary with placeholder values for all features
    features = {
        'Chroma': 0.0,
        'Tempo': 0.0,
        'Spectral_Centroid': 0.0,
        'Zero_Crossing_Rate': 0.0,
        'MFCC_1': 0.0, 'MFCC_2': 0.0, 'MFCC_3': 0.0, 'MFCC_4': 0.0,
        'MFCC_5': 0.0, 'MFCC_6': 0.0, 'MFCC_7': 0.0, 'MFCC_8': 0.0,
        'MFCC_9': 0.0, 'MFCC_10': 0.0, 'MFCC_11': 0.0, 'MFCC_12': 0.0,
        'MFCC_13': 0.0,
        'Rhythmic_Regularity': 0.0,
        'Placeholder_Feature': 0.0  # Add a placeholder feature for alignment
    }

    try:
        # Extract real features
        features['Chroma'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        features['Tempo'] = librosa.beat.tempo(y=y, sr=sr)[0]
        features['Spectral_Centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['Zero_Crossing_Rate'] = np.mean(librosa.feature.zero_crossing_rate(y=y))

        # Add MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'MFCC_{i+1}'] = np.mean(mfccs[i])

        # Add additional rhythmic features
        features['Rhythmic_Regularity'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

    except Exception as e:
        print(f"Error while extracting features: {e}. Using placeholder values for missing features.")

    print("Features extracted successfully.")
    return [features[col] for col in [
        'Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate',
        'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5',
        'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
        'MFCC_11', 'MFCC_12', 'MFCC_13', 'Rhythmic_Regularity',
        'Placeholder_Feature'  # Ensure alignment with training features
    ]]


def genre_classification(audio_path):
    """
    Classifies the genre of a song based on its audio file.

    Parameters:
        audio_path (str): Path to the audio file (.wav).

    Returns:
        str: The predicted genre (e.g., 'classical' or 'disco').
    """
    # Load the saved model, scaler, and label encoder
    print("Loading the saved model, scaler, and label encoder...")
    rf_classifier = joblib.load(r"C:\Users\Lenovo\Desktop\MusicMachineLearning\randomforest_2genres\rf_genre_classifier.pkl")
    scaler = joblib.load(r'C:\Users\Lenovo\Desktop\MusicMachineLearning\randomforest_2genres\scaler.pkl')
    label_encoder = joblib.load(r'C:\Users\Lenovo\Desktop\MusicMachineLearning\randomforest_2genres\label_encoder.pkl')

    # Extract features from the audio file
    print(f"Extracting features from audio: {audio_path}")
    input_features = extract_features(audio_path)

    # Convert the features into a 2D array for scaling
    input_features = np.array(input_features).reshape(1, -1)

    # Scale the features
    print("Scaling input features...")
    input_features_scaled = scaler.transform(input_features)

    # Predict the genre using the trained model
    print("Predicting the genre...")
    prediction = rf_classifier.predict(input_features_scaled)

    # Decode the predicted label back to the genre name
    predicted_genre = label_encoder.inverse_transform(prediction)[0]
    return predicted_genre


if __name__ == "__main__":
    # Path to the song
    song_path = r'C:\Users\Lenovo\Desktop\MusicMachineLearning\Remix_recom\01.wav'

    # Predict the genre
    print("Classifying the genre of the song...")
    predicted_genre = genre_classification(song_path)
    print(f"The predicted genre is: {predicted_genre}")
