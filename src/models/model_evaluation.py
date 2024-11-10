import logging
import pickle
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature
import json
import os

# Initialize DAGsHub and MLflow
dagshub_token = "10a7c2f7d30e69f138e738ca411a1fbd78583d48"
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri('https://dagshub.com/Kushagra-Bisht/Credit_Risk_Modelling_MLOPs.mlflow')

# Create a logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

# Create console and file handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data_for_evaluation():
    try:
        logger.debug("Starting the data loading process for evaluation.")
        test = pd.read_csv("data/interim/test.csv")
        X_test = test.drop(columns=['Approved_Flag'])
        y_test = test['Approved_Flag']
        logger.debug('Test data loaded successfully.')
        return X_test, y_test
    except Exception as e:
        logger.error("Error in loading test data: %s", e)
        raise

def load_model():
    try:
        logger.debug("Loading the pickled model.")
        with open('model/model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        logger.debug("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error("Error in loading the model: %s", e)
        raise

import os
import json

def save_model_info(run_id, model_name, file_path):
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    
    # Prepare the data to save
    model_info = {
        "run_id": run_id,
        "model_path": model_name,
        # Add any other relevant information you want to save
    }

    try:
        # Write the model information to the file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        print(f"Model information saved to {file_path}")
    except Exception as e:
        print(f"Failed to save model information: {e}")


def evaluate_model(model, X_test, y_test):
    try:
        logger.debug("Evaluating the model.")

        # Label encode y_test to ensure consistency with training
        label_encoder = LabelEncoder()
        y_test = label_encoder.fit_transform(y_test)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Log metrics to MLflow
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)

        # Log hyperparameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("random_state", 42)

        # Generate and log classification report
        report = classification_report(y_test, y_pred)
        print("\nClassification Report:")
        print(report)

        with open('classification_report.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report.txt')

        # Save and log test data
        X_test.to_csv("X_test.csv", index=False)
        y_test_series = pd.Series(y_test, name="Approved_Flag")
        y_test_series.to_csv("y_test.csv", index=False)
        mlflow.log_artifact("X_test.csv")
        mlflow.log_artifact("y_test.csv")

        # Log the model with the inferred signature
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

    except Exception as e:
        logger.error("Error in evaluating the model: %s", e)
        raise

def main():
    with mlflow.start_run() as run:
        X_test, y_test = load_data_for_evaluation()
        model = load_model()
        evaluate_model(model, X_test, y_test)
        save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

if __name__ == '__main__':
    main()
