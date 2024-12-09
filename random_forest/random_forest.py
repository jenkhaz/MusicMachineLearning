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

    # Create a custom colormap for the confusion matrix
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#fef5f8","#830131"], N=256)

    # Plot and save confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cmap=custom_cmap, cbar=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix with Custom Color Gradient')
    plt.savefig('confusion_matrix_custom_gradient.png')  # Save plot
    plt.show()

    # Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'max_features': [None, 'sqrt'],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        estimator=rf_classifier,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train_scaled, y_train)
    best_rf_classifier = grid_search.best_estimator_
    print("Best parameters found:")
    print(grid_search.best_params_)

    # Evaluate the optimized model
    print("Evaluating the optimized model...")
    y_pred_best = best_rf_classifier.predict(X_test_scaled)
    best_accuracy = accuracy_score(y_test, y_pred_best)
    print(f"Optimized Accuracy: {best_accuracy:.2f}")
    print("Optimized Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

    # Save models and encoders
    print("Saving the model, scaler, and label encoder...")
    joblib.dump(best_rf_classifier, 'rf_genre_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Models saved.")

    # Feature importance analysis
    print("Performing feature importance analysis...")
    importances = best_rf_classifier.feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    })
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

    # Custom colors for feature importance plot
    colors = ['#1d434e', '#4eb6b0', '#830131', '#a96d83'] * 5

    # Plot and save feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Importance', y='Feature',
        data=feature_importances.head(20),
        palette=colors[:20]
    )
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')  # Save plot
    plt.show()


if __name__ == "__main__":
    main()
