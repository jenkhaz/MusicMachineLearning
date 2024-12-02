from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from preprocess import send_features
from kmeansall import k_means_clustering


# def visualize_clusters_and_genres(data, labels, original_labels, n_clusters):
#     """
#     Visualize clusters and original genres on PCA-reduced data.

#     Parameters:
#         data (np.ndarray): Preprocessed and standardized features.
#         labels (np.ndarray): Cluster labels assigned by k-means.
#         original_labels (pd.Series): Original labels (e.g., genres) from the dataset.
#         n_clusters (int): Number of clusters.

#     Returns:
#         None: Displays two scatter plots.
#     """
#     # Step 1: Reduce data to 2D using PCA
#     pca = PCA(n_components=2)
#     reduced_data = pca.fit_transform(data)

#     # Step 2: Plot clusters based on k-means labels
#     plt.figure(figsize=(8, 6))
#     plt.scatter(
#         reduced_data[:, 0],
#         reduced_data[:, 1],
#         c=labels,
#         cmap='viridis',
#         s=50
#     )
#     plt.title(f'K-Means Clustering Visualization ({n_clusters} Clusters)')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.colorbar(label='Cluster Label')
#     plt.grid()
#     plt.show()

#     # Step 3: Plot data points colored by their original genre
#     unique_genres = original_labels.unique()
#     genre_colors = {genre: idx for idx, genre in enumerate(unique_genres)}
#     color_map = original_labels.map(genre_colors)

#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(
#         reduced_data[:, 0],
#         reduced_data[:, 1],
#         c=color_map,
#         cmap='tab10',
#         s=50
#     )
#     plt.title('Data Points Colored by Original Genre')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     colorbar = plt.colorbar(scatter, ticks=range(len(unique_genres)))
#     colorbar.ax.set_yticklabels(unique_genres)  # Map colors to genres
#     colorbar.set_label('Original Genre')
#     plt.grid()
#     plt.show()

def visualize_clusters_and_genres(data, labels, original_labels, n_clusters):
    """
    Visualize clusters and original genres on PCA-reduced data.

    Parameters:
        data (np.ndarray): Preprocessed and standardized features.
        labels (np.ndarray): Cluster labels assigned by k-means.
        original_labels (pd.Series): Original labels (e.g., genres) from the dataset.
        n_clusters (int): Number of clusters.

    Returns:
        None: Displays two scatter plots.
    """
    # Step 1: Reduce data to 2D using PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Step 2: Plot clusters based on k-means labels
    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=labels,
        cmap='viridis',
        s=50
    )
    print("PLOTTING FIRST GRAPH")
    plt.title(f'K-Means Clustering Visualization ({n_clusters} Clusters)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.grid()
    plt.show()  # Ensure this plot is displayed
    print("DONE PLOTTING FIRST")
    # Debugging: Check unique genres and their counts
    print("Unique genres:", original_labels.unique())
    print("Genre counts:", original_labels.value_counts())

    # Step 3: Plot data points colored by their original genre
    unique_genres = original_labels.unique()
    genre_colors = {genre: idx for idx, genre in enumerate(unique_genres)}
    print("Genre to Color Mapping:", genre_colors)  # Debug the color mapping
    color_map = original_labels.map(genre_colors)

    print("PLOTTING SECOND GRAPH")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=color_map,
        cmap='tab10',
        s=50
    )
    plt.title('Data Points Colored by Original Genre')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    colorbar = plt.colorbar(scatter, ticks=range(len(unique_genres)))
    colorbar.ax.set_yticklabels(unique_genres)  # Map colors to genres
    colorbar.set_label('Original Genre')
    plt.grid()
    plt.show()  # Ensure this plot is displayed

# Example Usage

preprocessed_features=send_features()
# Perform clustering
n_clusters = 10
clustering_results = k_means_clustering(preprocessed_features, n_clusters)


# Load your dataset from the uploaded file
file_path = 'C:\\Users\\user\\musicml_490\\MusicMachineLearning\\K-means-all\\audio_features_ALL.csv'
  # Adjust the path if needed
audio_features_all = pd.read_csv(file_path)

# Display the first few rows to confirm it's loaded correctly
print(audio_features_all.head())

# Extract cluster labels and original labels
cluster_labels = clustering_results['cluster_labels']
original_labels = audio_features_all['Label']  # Assuming 'Label' contains genres

# Visualize clusters and genres
visualize_clusters_and_genres(preprocessed_features, cluster_labels, original_labels, n_clusters)
