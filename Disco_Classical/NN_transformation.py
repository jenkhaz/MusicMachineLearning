import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the K-Means model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load the cluster labels
with open('cluster_labels.pkl', 'rb') as f:
    cluster_labels = pickle.load(f)

# Load and preprocess data
data = pd.read_csv('C:\\Users\\User\\OneDrive - American University of Beirut\\Desktop\\E3\\EECE 490\\MLproj\\ML_Codes\\MusicMachineLearning\\Disco_Classical\\audio_features(disco_classical).csv')

# Define actionable features
actionable_features = ['Tempo', 'Rhythmic_Regularity', 'Spectral_Centroid']  # Add more actionable features as needed
X = data[actionable_features]
y_labels = data['Label'] 

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans to find clusters
kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Map clusters to genres
cluster_labels = {cluster: data[data['Cluster'] == cluster]['Label'].mode()[0] for cluster in range(2)}
disco_cluster_label = [k for k, v in cluster_labels.items() if v == 'disco'][0]
classical_cluster_label = [k for k, v in cluster_labels.items() if v == 'classical'][0]

# Calculate cluster centroids
disco_centroid = kmeans.cluster_centers_[disco_cluster_label]
classical_centroid = kmeans.cluster_centers_[classical_cluster_label]

# Define target transformation vectors
target_transformations = []
for i, row in data.iterrows():
    if row['Cluster'] == classical_cluster_label:
        target_vector = disco_centroid - row[actionable_features].values
    else:
        target_vector = np.zeros(len(actionable_features))  # No transformation needed for disco
    target_transformations.append(target_vector)

# Add target transformations to the dataset
data['Target_Transformation'] = target_transformations

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target_transformations, test_size=0.2, random_state=42)

# Define the neural network
model = Sequential([
    Dense(64, input_dim=len(actionable_features), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(actionable_features), activation='linear')  # Output: transformation vector
])

# Custom loss function to encourage boundary crossing and cluster membership
def transformation_loss(y_true, y_pred):
    # y_true: Target transformation vector
    # y_pred: Predicted transformation vector
    transformed_point = y_pred + X_scaled  # Add predicted transformation to original features
    
    # Boundary crossing loss (assumes linear boundary)
    boundary_loss = tf.reduce_mean(tf.maximum(0.0, -tf.reduce_sum((disco_centroid - classical_centroid) * transformed_point, axis=1)))

    # Cluster radius loss
    distance_to_centroid = tf.norm(transformed_point - disco_centroid, axis=1)
    radius_loss = tf.reduce_mean(tf.maximum(0.0, distance_to_centroid - np.linalg.norm(disco_centroid - classical_centroid)))
    
    # Identity preservation loss (optional)
    identity_loss = tf.reduce_mean(tf.square(y_pred[:, 0]))  # Example: penalize changes to chroma if it's the first feature
    
    return boundary_loss + radius_loss + identity_loss

# Compile the model
model.compile(optimizer='adam', loss=transformation_loss, metrics=['mae'])

# Train the model
model.fit(X_train, np.array(y_train), epochs=50, batch_size=16, validation_data=(X_test, np.array(y_test)))

# Predict transformations
new_song_features = np.array([X_scaled[0]])  # Replace with actual features of a classical song
transformation_vector = model.predict(new_song_features)

# Apply the transformation
transformed_features = new_song_features + transformation_vector
transformed_cluster = kmeans.predict(transformed_features)

if transformed_cluster == disco_cluster_label:
    print("Successfully transformed into the disco cluster!")
else:
    print("Transformation unsuccessful. Needs refinement.")

# Generate recommendations
recommendations = []
for i, feature in enumerate(actionable_features):
    change = transformation_vector[0][i]
    if change > 0:
        recommendations.append(f"Increase {feature} by {change:.2f}.")
    elif change < 0:
        recommendations.append(f"Decrease {feature} by {-change:.2f}.")
print("Recommendations for transformation:", recommendations)
