import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.losses import MeanSquaredError
import logging
import pandas as pd

"""
This script processes audio files, extracts Mel spectrogram features, trains an autoencoder, and saves encoded features for analysis. It handles specific genres (classical and disco).
"""

logging.basicConfig(level=logging.INFO)

def extract_features(file_path, n_mels=128, max_length=130):
    """
    Extract Mel spectrogram features from an audio file.
    
    Args:
    file_path (str): Path to the audio file.
    n_mels (int): Number of Mel bands to generate.
    max_length (int): Maximum length for padding/truncation.

    Returns:
    np.ndarray: Transposed and normalized Mel spectrogram features.
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
        The encoder model
    """
    input_layer = layers.Input(shape=(input_dim,))  # Ensure input shape is a tuple
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(32, activation='relu', name='encoder_output')(x)

    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='sigmoid')(x)

    autoencoder = models.Model(input_layer, decoded)
    return autoencoder


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


def prepare_spectrogram_vectors(data_dir, output_csv_path="audio_features_spectrogram.csv"):
    """
    Prepare spectrogram features and save them to a CSV file.
    
    Args:
        data_dir (str): Path to dataset directory.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        tuple: Extracted labels and flattened features.
   
    """
    # Prepare the dataset
    features, labels = prepare_dataset(data_dir)

    # Flatten the features to match the autoencoder input requirements
    flattened_features = features.reshape(features.shape[0], -1)

    # Create a DataFrame for easy saving
    df = pd.DataFrame(flattened_features)
    df['label'] = labels  # Add labels as a separate column

    # Save to CSV
    output_csv_path = "audio_features_spectrogram.csv"  # Update with your desired path
    df.to_csv(output_csv_path, index=False)
    print(f"Features and labels saved to {output_csv_path}")

    return labels, flattened_features
    

def extract_features_from_csv_file(csv_file="audio_features_spectrogram.csv"):
    """Load features and labels from CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        tuple: Extracted labels, features, flattened_features.
    """
    
    data = pd.read_csv(csv_file)

    labels = data.iloc[:, -1]  # Extract the last column as labels
    features = data.iloc[:, :-1]  # Extract all other columns as features

    flattened_features = features.to_numpy()
  
    return labels, features, flattened_features


def train_autoencoder(flattened_features):
    """Train the autoencoder model.

    Args:
        flattened_features (np.ndarray): Flattened feature vectors.

    Returns:
        tuple: Input shape and the trained autoencoder model.
    """

    # Define the autoencoder and encoder
    input_shape = flattened_features.shape[1]  # Number of features per sample
    autoencoder = create_autoencoder(input_shape)

    # Prepare the training data
    X_train = flattened_features  # Use features as input and target
    y_train = flattened_features  # Target is the same as input

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the autoencoder
    history = autoencoder.fit(
        X_train, 
        y_train, 
        epochs=50,          # Number of training epochs
        batch_size=32,      # Batch size for training
        validation_split=0.2,  # Use 20% of the data for validation
        verbose=1           # Print training progress
    )

    # Save the trained autoencoder
    autoencoder_save_path = "trained_autoencoder.h5"
    autoencoder.save(autoencoder_save_path)
    print(f"Trained autoencoder saved to {autoencoder_save_path}")
    
    return input_shape, autoencoder


def recreate_encoder(input_dim, autoencoder):
    """Reconstruct the encoder model from a trained autoencoder.

    Args:
        input_dim (int): Flattened input size.

    Returns:
        Model: Reconstructed encoder model.
    """
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(32, activation='relu', name='encoder_output')(x)
    encoder = Model(input_layer, encoded)
    
    # Transfer weights from the autoencoder to the encoder
    for layer in encoder.layers:
        layer_name = layer.name
        if layer_name in [l.name for l in autoencoder.layers]:
            layer.set_weights(autoencoder.get_layer(layer_name).get_weights())

    print("Encoder reconstructed successfully.")
    
    encoder.save("trained_encoder.h5")
    print("Encoder saved successfully.")
    
    return encoder

def create_encoded_features(encoder, features, encoded_csv_file="trained_encoder_features.csv"):
    """Generate encoded features and save them to a CSV file.

    Args:
        encoder (Model): Trained encoder model.
        features (np.ndarray): Original feature set.
        encoded_csv_file (str): Path to save encoded features.

    Returns:
        pd.Dataframe: Dataframe of encoded features.
    """
   
    encoded_features = encoder.predict(features)
    print(f"Encoded features shape: {encoded_features.shape}")
    
    encoded_features_df = pd.DataFrame(encoded_features)
    encoded_features_df['label'] = labels

    encoded_features_df.to_csv(encoded_csv_file, index=False)
    print(f"Encoded features saved to {encoded_csv_file}")

    return encoded_features_df
# Load dataset
data_dir= "C:\\Users\\user\\musicml_490\\Segmented_Audio\\Segmented_Audio"

labels, features, flattened_features = extract_features_from_csv_file()

input_shape, autoencoder = train_autoencoder(flattened_features)

encoder = recreate_encoder(input_shape, autoencoder)

encoded_features_df = create_encoded_features(encoder, features)
