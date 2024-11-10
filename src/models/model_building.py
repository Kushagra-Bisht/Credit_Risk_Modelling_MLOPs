import os
import pickle
import logging
import pandas as pd
from imblearn.under_sampling import TomekLinks
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Initialize logger
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

# Create formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_and_split_data():
    # Your data loading and splitting logic
    logger.debug("Loading and splitting data.")
    # Example: Load dataset (replace with your actual data loading code)
    train = pd.read_csv("C:/Users/LCM/Desktop/Credit_Risk_Modelling/data/interim/train.csv")
    X_train = train.drop(columns=['Approved_Flag'])
    y_train = train['Approved_Flag']
    return X_train, y_train

def build_pipeline(X_train, y_train):
    try:
        logger.debug("Starting the model building process.")

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

        # Apply Label Encoding to the target variable (y_train)
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        logger.debug("Label encoding of target variable (y_train) done.")

        # Create a model pipeline with preprocessing
        model_pipeline = Pipeline(steps=[
            ('preprocessor', column_transformer),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42))
        ])

        # Fit the model pipeline before applying Tomek Links
        model_pipeline.fit(X_train, y_train_encoded)
        logger.debug("Model built and trained successfully.")

        # Create model folder if it doesn't exist
        model_folder = 'model'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            logger.debug(f"Created folder: {model_folder}")

        # Define the file path for saving the model
        model_filepath = os.path.join(model_folder, 'model.pkl')

        # Save the model using pickle
        with open(model_filepath, 'wb') as model_file:
            pickle.dump(model_pipeline, model_file)
        
        logger.debug(f"Model built and saved successfully at {model_filepath}.")
        return model_pipeline

    except Exception as e:
        logger.error("Error in building model: %s", e)
        raise

def main():
    # Load and split data
    X_train, y_train = load_and_split_data()

    # Build and save the model
    build_pipeline(X_train, y_train)

if __name__ == '__main__':
    main()
