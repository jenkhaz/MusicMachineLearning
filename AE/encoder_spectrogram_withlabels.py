import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model, Model
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

def extract_features(file_path, n_mels=128, max_length=130):
    """
    Extract Mel spectrogram features from an audio file.
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, duration=30)
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        # Pad or truncate
        if spectrogram_db.shape[1] > max_length:
            spectrogram_db = spectrogram_db[:, :max_length]
        else:
            spectrogram_db = np.pad(
                spectrogram_db, ((0, 0), (0, max_length - spectrogram_db.shape[1])), mode="constant"
            )
        spectrogram_db = (spectrogram_db - np.min(spectrogram_db)) / (np.max(spectrogram_db) - np.min(spectrogram_db) + 1e-8)
        return spectrogram_db.T  # Transpose
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

# Step 2: Build the Encoder-Decoder
def create_autoencoder(input_dim):
    """
    Create an autoencoder and an encoder model.

    Args:
        input_dim (int): Number of input features (flattened feature size).

    Returns:
        tuple: (autoencoder model, encoder model)
    """
    input_layer = layers.Input(shape=(input_dim,))  # Ensure input shape is a tuple
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(18, activation='relu', name='encoder_output')(x)

    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)
    return autoencoder, encoder


def prepare_dataset(data_dir, n_mels=128, max_length=130):
    """
    Prepare dataset by extracting features and labels from a list of directories.

    Args:
        data_dir (list): List of paths to directories containing genre subfolders.
        n_mels (int): Number of Mel bands for feature extraction.
        max_length (int): Maximum length for padding/truncation.

    Returns:
        np.ndarray: Array of extracted features.
        np.ndarray: Array of corresponding labels.
    """
    features = []
    labels = []

    for genre_folder in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre_folder)
        logging.info(f"Processing file: {genre_path}")
        
        if os.path.isdir(genre_path):
            
            for file in os.listdir(genre_path):
                if file.endswith(".mp3") or file.endswith(".wav"):
                    file_path = os.path.join(genre_path, file)
                    
                    try:
                        logging.info(f"Processing file: {file_path}")
                        feature = extract_features(file_path, n_mels, max_length)
                        if feature is not None:
                            features.append(feature)
                            labels.append(genre_folder)  # Use folder name as the label
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

    return np.array(features), np.array(labels)

# Load dataset
data_dir= "C:\\Users\\user\\musicml_490\\Segmented_Audio\\Segmented_Audio"

""" Uncomment the below code to create a dataset of spectrogram features using prepare dataset"""
# # Prepare the dataset
# features, labels = prepare_dataset(data_dir)

# # Flatten the features to match the autoencoder input requirements
# flattened_features = features.reshape(features.shape[0], -1)

# # Create a DataFrame for easy saving
# df = pd.DataFrame(flattened_features)
# # df['label'] = labels  # Add labels as a separate column

# # Save to CSV
# output_csv_path = "audio_features_spectrogram.csv"  # Update with your desired path
# df.to_csv(output_csv_path, index=False)
# print(f"Features and labels saved to {output_csv_path}")


""" Uncomment the below code to use an already existing dataset of spectrogram features"""

# csv_file = "audio_features_spectrogram.csv"  # Path to your CSV file
# data = pd.read_csv(csv_file)

# labels = data.iloc[:, -1]  # Extract the last column as labels
# features = data.iloc[:, :-1]  # Extract all other columns as features

# flattened_features = features.to_numpy()


# If you saved the encoder model:
# encoder = load_model("path_to_saved_encoder_model.h5")

"""Uncomment the below code to create the autoencoder and encoder. 
Highly recommended not to use the outputted encoder since it's not accurate"""
# # Define the autoencoder and encoder
# input_shape = flattened_features.shape[1]  # Number of features per sample
# autoencoder, encoder = create_autoencoder(input_shape)

# # Save the encoder model
# encoder_save_path = "encoder_model_18features.h5"
# encoder.save(encoder_save_path)
# print(f"Encoder saved to {encoder_save_path}")

# # Compile the autoencoder (necessary for training, even if not training here)
# autoencoder.compile(optimizer='adam', loss='mse')

# # Encode the features
# encoded_vectors = encoder.predict(flattened_features, batch_size=32)

# if len(encoded_vectors) != len(labels):
#     raise ValueError("Mismatch between encoded vectors and labels.")

# # Create a DataFrame for saving
# df_encoded = pd.DataFrame(encoded_vectors)
# df_encoded['label'] = labels  # Add labels as a separate column

# # Save to CSV
# encoded_csv_path = "encoded_audio_features_spectrogram_18features.csv"  # Update with your desired path
# df_encoded.to_csv(encoded_csv_path, index=False)
# print(f"Encoded vectors and labels saved to {encoded_csv_path}")

"""Uncomment below to train the autoencoder"""
# Prepare the training data
# X_train = flattened_features  # Use features as input and target
# y_train = flattened_features  # Target is the same as input

# # Compile the autoencoder
# autoencoder.compile(optimizer='adam', loss='mse')

# # Train the autoencoder
# history = autoencoder.fit(
#     X_train, 
#     y_train, 
#     epochs=50,          # Number of training epochs
#     batch_size=32,      # Batch size for training
#     validation_split=0.2,  # Use 20% of the data for validation
#     verbose=1           # Print training progress
# )

# # Save the trained autoencoder
# autoencoder_save_path = "trained_autoencoder.h5"
# autoencoder.save(autoencoder_save_path)
# print(f"Trained autoencoder saved to {autoencoder_save_path}")


