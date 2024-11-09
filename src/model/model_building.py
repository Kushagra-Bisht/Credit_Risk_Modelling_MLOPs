import logging
import yaml
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from imblearn.under_sampling import TomekLinks

# Create a logger
logger = logging.getLogger("data_transformation")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel(logging.ERROR)

# Create formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load and split data
def load_and_split_data():
    try:
        logger.debug("Starting the data loading process.")
        train = pd.read_csv("C:/Users/LCM/Desktop/Credit_Risk_Modelling/data/interim/train.csv")
        test = pd.read_csv("C:/Users/LCM/Desktop/Credit_Risk_Modelling/data/interim/test.csv")
        logger.debug('Data loaded properly')

        X_train = train.drop(columns=['Approved_Flag'])
        y_train = train['Approved_Flag']
        X_test = test.drop(columns=['Approved_Flag'])
        y_test = test['Approved_Flag']
        
        logger.debug('Train and test data split into X and y')
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error("Error in loading and splitting data: %s", e)
        raise

# Data transformation
def transform_data(X_train, X_test):
    try:
        logger.debug("Starting the data transformation process.")
        
        # Identify numeric and categorical columns
        num = X_train.select_dtypes(include=['int64', 'float64']).columns
        cat = X_train.select_dtypes(include=['O']).columns

        # Set up transformers for categorical and numeric features
        categorical_transformer = Pipeline(steps=[("OHE", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        numeric_transformer = Pipeline(steps=[('Scaling', StandardScaler())])

        # Apply transformations to both numeric and categorical features
        column_transformer = ColumnTransformer(transformers=[
            ('num', numeric_transformer, num),
            ('cat', categorical_transformer, cat)
        ], remainder='passthrough')

        X_train_transformed = column_transformer.fit_transform(X_train)
        X_test_transformed = column_transformer.transform(X_test)

        logger.debug("Data transformed successfully")
        return X_train_transformed, X_test_transformed

    except Exception as e:
        logger.error("Error in transforming data: %s", e)
        raise

# Label encoding for target variable
def encode_labels(y_train, y_test):
    try:
        logger.debug("Starting label encoding.")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        logger.debug("Labels encoded successfully.")
        return y_train_encoded, y_test_encoded
    except Exception as e:
        logger.error("Error in label encoding: %s", e)
        raise

# Apply Tomek links for balancing
def apply_tomek_links(X_train, y_train):
    try:
        logger.debug("Applying Tomek Links for resampling.")
        tomek_links = TomekLinks()
        X_resampled, y_resampled = tomek_links.fit_resample(X_train, y_train)
        logger.debug("Tomek Links applied successfully.")
        return X_resampled, y_resampled
    except Exception as e:
        logger.error("Error in applying Tomek Links: %s", e)
        raise

# Train the model
def train_model(X_train, y_train):
    try:
        logger.debug("Training the RandomForest model.")
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_train, y_train)
        logger.debug("Model trained successfully.")
        return model
    except Exception as e:
        logger.error("Error in training model: %s", e)
        raise

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    try:
        logger.debug("Evaluating the model.")
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Print results
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        logger.debug("Model evaluation completed successfully.")
    except Exception as e:
        logger.error("Error in evaluating model: %s", e)
        raise

# Main function to execute the workflow
def main():
    try:
        X_train, X_test, y_train, y_test = load_and_split_data()
        X_train_transformed, X_test_transformed = transform_data(X_train, X_test)
        y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)
        X_resampled, y_resampled = apply_tomek_links(X_train_transformed, y_train_encoded)
        model = train_model(X_resampled, y_resampled)
        evaluate_model(model, X_test_transformed, y_test_encoded)
        evaluate_model(model, X_train_transformed, y_train_encoded)
    except Exception as e:
        logger.error('Failed to complete the process: %s', e)

if __name__ == '__main__':
    main()

