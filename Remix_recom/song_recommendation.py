import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_songs(input_song_name, csv_path, top_n=5):
    """
    Recommends songs from the dataset that would go well with the input song for a remix.

    Parameters:
    - input_song_name (str): Name of the song (must match the Segment_File_Name in the CSV).
    - csv_path (str): Path to the CSV file containing song features.
    - top_n (int): Number of recommendations to return.

    Returns:
    - List of recommended songs.
    """
    # Load the dataset
    data = pd.read_csv(csv_path)

    # Ensure the required columns are present
    if 'Segment_File_Name' not in data.columns:
        raise ValueError("The dataset must contain a 'Segment_File_Name' column.")

    # Extract the base song name (remove segment info)
    def get_base_song_name(file_name):
        return "_".join(file_name.split("_segment")[0].split())

    input_base_song_name = get_base_song_name(input_song_name)

    # Filter out segments of the same song
    data['Base_Song_Name'] = data['Segment_File_Name'].apply(get_base_song_name)
    filtered_data = data[data['Base_Song_Name'] != input_base_song_name]

    # Drop non-numeric columns for similarity calculation
    non_feature_columns = ['Label', 'Segment_File_Name', 'Base_Song_Name']
    feature_columns = [col for col in data.columns if col not in non_feature_columns]

    # Convert feature columns to numeric values (handle nested lists or complex strings)
    for col in feature_columns:
        filtered_data[col] = filtered_data[col].apply(
            lambda x: np.mean([float(i) for i in str(x).strip("[]").split(",") if i.strip()]) if isinstance(x, str) else x
        )
        data[col] = data[col].apply(
            lambda x: np.mean([float(i) for i in str(x).strip("[]").split(",") if i.strip()]) if isinstance(x, str) else x
        )

    # Ensure the input song exists in the dataset
    if input_song_name not in data['Segment_File_Name'].values:
        raise ValueError(f"The song '{input_song_name}' is not in the dataset.")

    # Normalize features
    features = filtered_data[feature_columns]
    features_normalized = (features - features.mean()) / features.std()

    # Extract and normalize the input song features
    input_features = data.loc[data['Segment_File_Name'] == input_song_name, feature_columns]
    input_features = input_features.astype(float)  # Ensure all values are numeric
    input_features_normalized = (input_features - features.mean()) / features.std()
    input_features_normalized = input_features_normalized.to_numpy()  # Convert to NumPy array

    # Calculate similarity between the input song and all other songs
    similarity_scores = cosine_similarity(input_features_normalized, features_normalized).flatten()

    # Get the indices of the top_n most similar songs
    similar_indices = np.argsort(similarity_scores)[::-1][:top_n]

    # Fetch recommended songs
    recommended_songs = filtered_data.iloc[similar_indices]['Segment_File_Name'].values

    return recommended_songs

# Example Usage
if __name__ == "__main__":
    # Input CSV file path
    csv_file = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\audio_features_genres_with_segments.csv"
    
    # Input song name
    input_song = "Alexander Scriabin, Julius Asal - PROLOGUE (Quasi Niente from Piano Sonata No. 1 in F Minor, Op. 6- IV. Funebre) - Upright Version_segment_0_30.wav"

    # Get recommendations
    try:
        recommendations = recommend_songs(input_song, csv_file, top_n=5)
        print(f"Songs recommended for remixing with '{input_song}':")
        for song in recommendations:
            print(f"- {song}")
    except ValueError as e:
        print(e)
