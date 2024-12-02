from sklearn.cluster import KMeans
from preprocess import send_features
import matplotlib.pyplot as plt

def elbow_method(data, max_clusters=20): #There are 10 genres originally
    """
    Perform the Elbow Method to determine the optimal number of clusters.

    Parameters:
        data (np.ndarray): Preprocessed and standardized dataset.
        max_clusters (int): Maximum number of clusters to test.

    Returns:
        None: Displays the Elbow plot.
    """
    # List to store inertia (sum of squared distances to centroids)
    inertia = []

    # Fit k-means for different values of k
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()

preprocessed_features_all = send_features()
# Run the Elbow Method on the preprocessed dataset
elbow_method(preprocessed_features_all, max_clusters=10)
