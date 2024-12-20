import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import librosa
import joblib

# Load pre-trained models and data
with open('kmeans_model_2_clusters.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('scaler_2_clusters.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('cluster_labels_2_clusters.pkl', 'rb') as f:
    cluster_labels = pickle.load(f)

# Load Random Forest model
rf_classifier = joblib.load('rf_genre_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def extract_features(y, sr):
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        rhythmic_regularity = tempo / 60

        features = [chroma, tempo, spectral_centroid, zcr, *mfccs, rhythmic_regularity]
        return [float(f) for f in features]
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

def transform_genre(song_path, target_genre='disco', step_size=0.5, max_steps=100):
    # Load audio
    y, sr = librosa.load(song_path, sr=22050)
    song_features = extract_features(y, sr)

    # Validate extracted features
    if not isinstance(song_features, list) or not all(isinstance(f, (int, float, np.float64)) for f in song_features):
        raise ValueError("Extracted features are invalid or contain non-scalar values.")
    
    # Convert to NumPy array
    song_features = np.array(song_features, dtype=np.float64)
    
    # Match expected feature count
    expected_feature_count = len(scaler.mean_)
    if len(song_features) < expected_feature_count:
        song_features = np.append(song_features, [0] * (expected_feature_count - len(song_features)))
    elif len(song_features) > expected_feature_count:
        song_features = song_features[:expected_feature_count]

    # Add the initial label placeholder for "classical"
    classical_label_numeric = label_encoder.transform(['classical'])[0]
    song_features = np.append(song_features, classical_label_numeric)

    # Scale the features
    scaled_features = scaler.transform([song_features[:-1]])[0]  # Exclude placeholder from scaling
    scaled_features = np.append(scaled_features, song_features[-1])  # Re-attach placeholder

    # Get disco cluster centroid from KMeans
    disco_centroid = kmeans.cluster_centers_[list(cluster_labels.values()).index(target_genre)]
    if len(disco_centroid) != len(scaled_features[:-1]):  # Ignore placeholder during comparison
        raise ValueError("Centroid feature count does not match scaled feature count.")
    
    # Adjust features towards centroid
    important_indices = [0, 1, 17]
    current_features = scaled_features.copy()

    # Define a threshold to check prediction stability
    no_change_threshold = 5  # Number of steps without prediction change
    no_change_count = 0
    previous_prediction = 'classical'
    step_size = 0.5  # Set your initial step size

    for step in range(max_steps):
        # Adjust important features towards the disco centroid
        for idx in important_indices:
            diff = disco_centroid[idx] - current_features[idx]
            adjustment = np.sign(diff) * step_size  # Move in the direction of the centroid
            if abs(adjustment) > abs(diff):  # Avoid overshooting
                adjustment = diff
            current_features[idx] += adjustment
        
        # Prepare the prediction input with the updated label placeholder
        prediction_input = current_features.reshape(1, -1)
        current_prediction_numeric = rf_classifier.predict(prediction_input)[0]
        current_prediction = label_encoder.inverse_transform([current_prediction_numeric])[0]

        # Check if the prediction remains unchanged
        if current_prediction == previous_prediction:
            no_change_count += 1
        else:
            no_change_count = 0  # Reset if prediction changes
        
        previous_prediction = current_prediction

        # Log the step and prediction
        print(f"Step {step + 1}, Adjusted Features: {current_features}, Prediction: {current_prediction}")

        # Break loop if target genre is reached
        if current_prediction == target_genre:
            print(f"Target genre '{target_genre}' reached at step {step + 1}.")
            break

        # Dynamically increase step size if prediction doesn't change
        if no_change_count >= no_change_threshold:
            step_size *= 1.5
            print(f"No change in prediction for {no_change_threshold} steps. Increasing step size to {step_size:.2f}.")

    print(f"Maximum steps reached. Final prediction: {current_prediction}")
    return current_features[:-1]  # Return features without the placeholder


# Example usage
song_path = r"C:\Users\User\OneDrive - American University of Beirut\Desktop\E3\EECE 490\MLproj\Segmented_Audio\Segmented_Audio\classical\01.wav"
y, sr = librosa.load(song_path, sr=22050)
song_features = extract_features(y, sr)
classical_label_numeric = label_encoder.transform(['classical'])[0]

final_features = transform_genre(song_path, target_genre='disco')
song_features = np.append(song_features, classical_label_numeric)

    # Scale the features
scaled_features = scaler.transform([song_features[:-1]])[0]  # Exclude placeholder from scaling
scaled_features = np.append(scaled_features, song_features[-1])  # Re-attach placeholder
print("\nFinal Adjusted Features:")
print(final_features)



def generate_recommendations(initial_features, final_features, feature_names):
    # Align feature lengths
    if len(initial_features) != len(final_features):
        print(f"Aligning features: initial ({len(initial_features)}) vs final ({len(final_features)})")
        min_length = min(len(initial_features), len(final_features))
        initial_features = initial_features[:min_length]
        final_features = final_features[:min_length]
    
    
    differences = np.array(final_features) - np.array(initial_features)
    recommendations = []

    for i, diff in enumerate(differences):
        if diff > 0:
            recommendations.append(f"Increase {feature_names[i]} by {diff:.2f}")
        elif diff < 0:
            recommendations.append(f"Decrease {feature_names[i]} by {abs(diff):.2f}")

    return recommendations


# Define feature names
feature_names = [
    "Chroma",
    "Tempo",
    "Spectral Centroid",
    "Zero Crossing Rate",
    "MFCC 1", "MFCC 2", "MFCC 3", "MFCC 4", "MFCC 5", "MFCC 6", "MFCC 7",
    "MFCC 8", "MFCC 9", "MFCC 10", "MFCC 11", "MFCC 12", "MFCC 13",
    "Rhythmic Regularity"
]


# Denormalize features back to original scale
initial_features_original = scaler.inverse_transform([scaled_features[:-1]])[0]  # Exclude placeholder
final_features_original = scaler.inverse_transform([final_features])  # No placeholder included in final_features

# Generate recommendations in the original scale
recommendations = generate_recommendations(initial_features_original, final_features_original[0], feature_names)

# Print recommendations
print("\nRecommendations for transitioning to the target genre (Original Scale):")
for rec in recommendations:
    print(rec)
