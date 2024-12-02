"""
The purpose is to make another CSV file to remove "Label".
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_audio_features_from_csv(file_path):
    """
    Preprocess the audio_features dataset for clustering from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing audio features.

    Returns:
        np.ndarray: Preprocessed and standardized features ready for clustering.
    """
    # Step 1: Load the dataset
    dataframe = pd.read_csv(file_path)
    
    # Step 2: Drop the 'Label' column
    features = dataframe.drop(columns=['Label'], errors='ignore')
    
    # Step 3: Convert list-like 'Tempo' feature into a scalar (mean value)
    if 'Tempo' in features.columns:
        features['Tempo'] = features['Tempo'].apply(lambda x: np.mean(eval(x)) if isinstance(x, str) else x)
    
    # Step 4: Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features


def send_features():
    # Example usage
    file_path = "C:\\Users\\user\\musicml_490\\MusicMachineLearning\\audio_features.csv"
    preprocessed_features = preprocess_audio_features_from_csv(file_path)

    print("Scaled features are: ", preprocessed_features)
    
    return preprocessed_features
    