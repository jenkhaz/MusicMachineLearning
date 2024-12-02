import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def preprocess_audio_features(file_path):
    """
    Preprocess the audio_features dataset for clustering.

    Parameters:
        dataframe (pd.DataFrame): The raw dataset.

    Returns:
        np.ndarray: Preprocessed and standardized features.
    """
    dataframe = pd.read_csv(file_path)
    # Step 1: Drop the 'Label' column
    features = dataframe.drop(columns=['Label'], errors='ignore')

    # Step 2: Convert list-like 'Tempo' feature into a scalar (mean value)
    if 'Tempo' in features.columns:
        features['Tempo'] = features['Tempo'].apply(lambda x: np.mean(eval(x)) if isinstance(x, str) else x)

    # Step 3: Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features

def send_features(): 
    # Preprocess the uploaded dataset
    preprocessed_features_all = preprocess_audio_features('C:\\Users\\user\\musicml_490\\audio_features_ALL.csv')
    return preprocessed_features_all