import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset_path = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\audio_features_genres_with_segments.csv"
data = pd.read_csv(dataset_path)

# Define feature columns
feature_columns = [
    'Chroma', 'Tempo', 'Spectral_Centroid', 'Zero_Crossing_Rate',
    'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6',
    'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12',
    'MFCC_13', 'Rhythmic_Regularity'
]

# Clean feature columns
for column in feature_columns:
    data[column] = data[column].apply(lambda x: float(x.strip('[]')) if isinstance(x, str) else x)

# Handle missing or invalid values
data[feature_columns] = data[feature_columns].fillna(data[feature_columns].mean())

# Preprocess features (scaling)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[feature_columns])

# Apply K-Means to create initial clusters
kmeans = KMeans(n_clusters=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Train a Random Forest model using the K-Means clusters as labels
X = scaled_features  # Input features
y = data['Cluster']  # K-Means cluster labels as target

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Function to extract features from an audio file
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract features
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo.item()
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1).tolist()
        
        rhythmic_regular = np.std(librosa.onset.onset_strength(y=y, sr=sr))
        
        features = [chroma, tempo, spectral_centroid, zero_crossing_rate] + mfcc_means + [rhythmic_regular]
        
        return np.array(features, dtype=np.float64)
    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        raise

# Function to recommend songs using Random Forest
def recommend_songs_random_forest(input_song_file, data, feature_columns, top_n=5):
    input_features = extract_features(input_song_file)
    input_features_scaled = scaler.transform([input_features])
    
    # Predict cluster using Random Forest
    predicted_cluster = rf.predict(input_features_scaled)[0]
    print(f"Predicted Cluster for {input_song_file}: {predicted_cluster}")
    
    # Filter songs from the predicted cluster
    cluster_songs = data[data['Cluster'] == predicted_cluster].copy()
    
    # Compute similarity within the cluster
    similarities = cosine_similarity(input_features_scaled, cluster_songs[feature_columns].values).flatten()
    cluster_songs['Similarity'] = similarities
    
    # Sort by similarity and filter unique songs
    cluster_songs = cluster_songs.sort_values(by='Similarity', ascending=False)
    cluster_songs['Base_Song_Name'] = cluster_songs['Segment_File_Name'].str.extract(r'([^_]+)')[0]
    unique_songs = cluster_songs.drop_duplicates(subset='Base_Song_Name')
    
    return unique_songs.head(top_n), similarities  # Make sure both recommendations and similarities are returned

# Example input songs
input_song_file_1 = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\Remix_recom\01.wav"
input_song_file_2 = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\Remix_recom\01 copy.wav"

# Get recommendations and similarities for Song 1
recommendations_1, similarities_1 = recommend_songs_random_forest(input_song_file_1, data, feature_columns)

# Get recommendations and similarities for Song 2
recommendations_2, similarities_2 = recommend_songs_random_forest(input_song_file_2, data, feature_columns)

print("\nRecommendations for Song 1:")
print(recommendations_1[['Segment_File_Name', 'Label']])

print("\nRecommendations for Song 2:")
print(recommendations_2[['Segment_File_Name', 'Label']])

# Ensure that similarities are aligned with the top 5 recommended songs
top_5_similarities_1 = similarities_1[:5]  # Take the top 5 similarities
top_5_songs_1 = recommendations_1['Segment_File_Name'].head(5)  # Take the top 5 song names

# Create the similarity bar plot for Song 1
plt.figure(figsize=(8, 6))
sns.barplot(x=top_5_similarities_1, y=top_5_songs_1, color="#4eb6b0")
plt.title("Similarity Scores for Recommended Songs (Song 1)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Songs")
plt.savefig("similarity_scores_plot_song_1.png")
plt.show()
