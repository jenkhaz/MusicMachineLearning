import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the dataset
csv_file_path = r"C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\audio_features_genres_with_segments.csv"  # Replace with your actual CSV file path
data = pd.read_csv(csv_file_path)

# Step 2: Preprocess the data
features = data.drop(columns=['Label', 'Segment_F', 'Rhythmic_'])  # Drop non-numeric columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
data['Cluster'] = clusters

# Step 4: Identify which cluster is classical and which is disco
classical_cluster = data[data['Label'] == 'classical']['Cluster'].mode()[0]
disco_cluster = 1 - classical_cluster  # Assuming only 2 clusters

# Step 5: Function to recommend disco songs for remixing
def recommend_songs(input_song_features, data, classical_cluster, disco_cluster, scaler):
    input_scaled = scaler.transform([input_song_features])  # Scale input features
    disco_songs = data[data['Cluster'] == disco_cluster]
    
    # Compute similarity
    disco_features = scaler.transform(disco_songs.drop(columns=['Label', 'Segment_F', 'Rhythmic_', 'Cluster']))
    similarities = cosine_similarity(input_scaled, disco_features).flatten()
    
    # Get top recommendations
    disco_songs['Similarity'] = similarities
    recommendations = disco_songs.sort_values(by='Similarity', ascending=False).head(5)
    return recommendations[['Label', 'Similarity']]

# Example input: A classical song's features (replace with actual feature values)
input_song = data[data['Label'] == 'classical'].iloc[0].drop(['Label', 'Segment_F', 'Rhythmic_', 'Cluster']).values

# Get recommendations
recommendations = recommend_songs(input_song, data, classical_cluster, disco_cluster, scaler)
print(recommendations)
