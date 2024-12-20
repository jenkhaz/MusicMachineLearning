import numpy as np
import pandas as pd
import joblib
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def main():
    # Load the dataset from the CSV file
    csv_file_path = r'C:\Users\Lenovo\Desktop\MusicMachineLearning\add_data\audio_features_genres_with_segments.csv'
    print("Loading dataset from CSV file...")
    data = pd.read_csv(csv_file_path)
    print(f"Dataset loaded. Total samples: {len(data)}")

    # Filter dataset to include only "classical" and "disco" genres
    genres_of_interest = ['classical', 'disco']
    data = data[data['Label'].isin(genres_of_interest)]
    print(f"Filtered dataset. Samples for classical and disco: {len(data)}")

    # Identify feature and label columns
    feature_columns = data.columns[:-1]  # All columns except the last one (features)
    label_column = data.columns[-1]      # Last column (genre labels)

    # Convert feature columns to numeric values
    print("Converting feature columns to numeric values...")
    def convert_to_float(x):
        try:
            if isinstance(x, (float, int)):
                return x
            elif isinstance(x, str):
                x_clean = re.sub(r'[^\d.\-]', '', x)
                return float(x_clean)
            else:
                return np.nan
        except ValueError:
            return np.nan

    for col in feature_columns:
        data[col] = data[col].apply(convert_to_float)

    # Drop rows with NaN values
    print("Dropping rows with NaN values...")
    initial_sample_count = len(data)
    data.dropna(subset=feature_columns, inplace=True)
    final_sample_count = len(data)
    print(f"Dropped {initial_sample_count - final_sample_count} rows due to NaN values. Total samples now: {final_sample_count}")

    # Separate features and labels
    X = data[feature_columns].values
    y = data[label_column].values

    # Encode the labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest classifier
    print("Training the Random Forest classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = rf_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Define a custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#fef5f8", "#830131"], N=256)

    # Plot and save confusion matrix with the custom colormap
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cmap=custom_cmap, cbar=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Save models and encoders
    print("Saving the model, scaler, and label encoder...")
    joblib.dump(rf_classifier, 'rf_genre_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Models saved.")

if __name__ == "__main__":
    main()
