import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def drop_genres(dataframe, genre):
    """
    Drops rows from the dataframe where the label matches any genre in the given array.
    This is used in case of wanting to cluster less than the 4 genres included in the csv.
    
    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        genre (list): List of genres to drop.

    Returns:
        pd.DataFrame: The dataframe with specified genres dropped.
    """
    return dataframe[~dataframe['label'].isin(genre)]

def creating_kmeans_model(data, n_clusters, kmeans_model_path="kmeans_autoencoder.pkl"):

    """
    Creates and trains a KMeans clustering model, then maps predicted clusters to true labels.

    Parameters:
        data (pd.DataFrame): The input dataframe with features and labels.
        n_clusters (int): The number of clusters for the KMeans model.
        kmeans_model_path (str): File path to save the trained KMeans model.

    Returns:
        tuple: (numeric_labels, features, predicted_clusters, mapped_labels, true_label_mapping)
        - numeric_labels (np.array): True labels mapped to numeric values.
        - features (np.array): Feature matrix extracted from the dataframe.
        - predicted_clusters (np.array): Cluster predictions from the KMeans model.
        - mapped_labels (np.array): Predicted clusters mapped to true labels.
        - true_label_mapping (dict): Mapping of original labels to numeric values.
    """
    
    # Separate features and true labels
    true_labels = data['label']  # Assuming the last colu   mn is 'label'
    print("True Labels:\n", true_labels)
    features = data.drop(columns=['label']).to_numpy()

    # Map true labels to numeric values
    true_label_mapping = {label: idx for idx, label in enumerate(true_labels.unique())}
    numeric_labels = true_labels.map(true_label_mapping).to_numpy()
        
    # Train a new KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    
    # Predict clusters using the new KMeans model
    predicted_clusters = kmeans.predict(features)
    
    # Save the newly trained KMeans model
    kmeans_model_path = "C:\\Users\\user\\musicml_490\\MusicMachineLearning\\AE\\final_version\\kmeans_model_2_clusters.pkl"
    with open(kmeans_model_path, 'wb') as file:
        pickle.dump(kmeans, file)
    print(f"Trained KMeans model saved to {kmeans_model_path}.")

    # Map clusters to true labels using the Hungarian Algorithm
    cost_matrix = np.zeros((n_clusters, n_clusters))  # Assuming n_clusters clusters
    for i in range(n_clusters):  # Iterate over clusters
        for j in range(n_clusters):  # Iterate over true labels
            cost_matrix[i, j] = -np.sum((predicted_clusters == i) & (numeric_labels == j))  # Negative for Hungarian algorithm

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_label_mapping = {row: col for row, col in zip(row_ind, col_ind)}
    
    # Map predicted clusters to true labels
    mapped_labels = np.array([cluster_to_label_mapping[cluster] for cluster in predicted_clusters])
  
    return numeric_labels, features, predicted_clusters, mapped_labels, true_label_mapping

def accuracy(numeric_labels, mapped_labels):
    """
    Calculates the accuracy of clustering by comparing true labels to mapped clusters.

    Parameters:
        numeric_labels (np.array): Array of true labels in numeric format.
        mapped_labels (np.array): Array of predicted labels mapped to true labels.

    Returns:
        float: Accuracy of the clustering (as a percentage).
    """
    accuracy = accuracy_score(numeric_labels, mapped_labels)
    print(f"Clustering Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def visualization(n_clusters, features, predicted_clusters, true_label_mapping, accuracy):
    """
    Visualizes clusters and true labels using PCA for dimensionality reduction.

    Parameters:
        n_clusters (int): Number of clusters to visualize.
        features (np.array): Feature matrix of the data.
        predicted_clusters (np.array): Predicted clusters from the KMeans model.
        true_label_mapping (dict): Mapping of true labels to numeric values.
        accuracy (float): Accuracy of the clustering.

    Returns:
        None
    """ 
    # Visualize clusters with true labels
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    custom_colors = ['#1d434e', '#4eb6b0', '#830131', '#a96d83']

    # Create a scatter plot for clusters without labels
    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        plt.scatter(
            reduced_features[predicted_clusters == cluster, 0],
            reduced_features[predicted_clusters == cluster, 1],
            color=custom_colors[cluster],  # Use the custom color for each cluster
            label=f'Cluster {cluster}'
        )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title("KMeans Clustering Visualization (Unlabeled)")
    plt.legend()
    plt.grid()
    plt.show()

    # Visualize clusters with true labels
    plt.figure(figsize=(10, 8))
    for idx, label in enumerate(np.unique(numeric_labels)):
        plt.scatter(
            reduced_features[numeric_labels == label, 0],
            reduced_features[numeric_labels == label, 1],
            color=custom_colors[idx],  # Use the custom color for each label
            label=f'True Label {label} ({list(true_label_mapping.keys())[list(true_label_mapping.values()).index(label)]})'
        )

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f"True Labels Visualization with Accuracy {accuracy * 100:.2f}%")
    plt.legend()
    plt.grid()
    plt.show()

"""In this section, we are creating a K-mean model to cluster 4 genres, and another to cluster 2 genres (classical and disco)"""  
# Load the encoded features
encoded_csv_path = "C:\\Users\\user\\musicml_490\\from_trained_encoder_audio_features.csv"
data = pd.read_csv(encoded_csv_path)
genre = ["rock", "techno"]

# for 4 genres 
numeric_labels, features, predicted_clusters, mapped_labels, true_label_mapping = creating_kmeans_model(data, 4, "kmeans_autoencoder_4genres")
accuracy_value1 = accuracy(numeric_labels, mapped_labels)

visualization(4, features, predicted_clusters, true_label_mapping, accuracy_value1)

# for 2 genres
data = drop_genres(data, genre)
numeric_labels, features, predicted_clusters, mapped_labels, true_label_mapping = creating_kmeans_model(data, 2, "kmeans_autoencoder_2genres")
accuracy_value2 = accuracy(numeric_labels, mapped_labels)

visualization(2, features, predicted_clusters, true_label_mapping, accuracy_value2)
