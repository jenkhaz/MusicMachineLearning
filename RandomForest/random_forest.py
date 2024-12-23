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

def main():
    # Load the dataset from the CSV file
    csv_file_path = r'C:\Users\User\OneDrive - American University of Beirut\Desktop\E3\EECE 490\MLproj\ML_Codes\MusicMachineLearning\add_data\audio_features_genres_with_segments.csv'
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
            # If x is already a float or int, return it
            if isinstance(x, (float, int)):
                return x
            # If x is a string, remove any non-numeric characters except '.' and '-'
            elif isinstance(x, str):
                # Remove brackets and extra characters
                x_clean = re.sub(r'[^\d\.-]', '', x)
                return float(x_clean)
            else:
                # Handle any other types
                return np.nan
        except Exception as e:
            print(f"Error converting value '{x}': {e}")
            return np.nan  # Return NaN for problematic entries

    # Apply the conversion to all feature columns
    for col in feature_columns:
        data[col] = data[col].apply(convert_to_float)
    
    # Drop rows with NaN values in any feature columns
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
    
    # Initialize and train the Random Forest classifier
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
    
    # Plot confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cmap='#1d4a4e')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Hyperparameter tuning with Grid Search
    print("Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['auto', 'sqrt'],
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
    
    # Save the model, scaler, and label encoder
    print("Saving the model, scaler, and label encoder...")
    joblib.dump(best_rf_classifier, 'rf_genre_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Models saved.")
    
    print("Performing feature importance analysis...")
    importances = best_rf_classifier.feature_importances_
    feature_names = feature_columns  # Already defined earlier
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    
    # Plot the top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20), palette='viridis')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.show()
    joblib.dump(best_rf_classifier, 'rf_genre_classifier.pkl')

    
if __name__ == "_main_":
    main()