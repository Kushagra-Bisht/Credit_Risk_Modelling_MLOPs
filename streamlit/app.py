import streamlit as st
import pandas as pd
import mlflow
import dagshub
import logging
import joblib  # To load the LabelEncoder
import os 

# Logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize DagsHub and MLflow
dagshub_token = "10a7c2f7d30e69f138e738ca411a1fbd78583d48"
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri('https://dagshub.com/Kushagra-Bisht/Credit_Risk_Modelling_MLOPs.mlflow')

model_name = "model" 
model_version = 3
model_uri = f'models:/{model_name}/{model_version}'

# Cache the model and label encoder loading
@st.cache_resource
def load_model():
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error("Error loading the model. Please check the model URI and MLflow setup.")
        return None

@st.cache_resource
def load_label_encoder():
    try:
        label_encoder = joblib.load('models/label_encoder.pkl')
        logger.info("LabelEncoder loaded successfully.")
        return label_encoder
    except Exception as e:
        logger.error(f"Error loading LabelEncoder: {e}")
        st.error("Error loading the LabelEncoder. Please check the file path.")
        return None

# Load the model and label encoder once and reuse
model = load_model()
label_encoder = load_label_encoder()

if model is None or label_encoder is None:
    st.stop()  # Stop if the model or LabelEncoder didn't load

# Streamlit interface
st.title("Credit Risk Modelling")
st.write("Enter the features to predict the creditworthiness of an individual")

# Selected features for prediction
selected_features = [
    'max_recent_level_of_deliq', 'num_std_12mts', 'time_since_recent_payment', 
    'enq_L3m', 'PL_enq_L12m', 'pct_of_active_TLs_ever', 'pct_PL_enq_L6m_of_ever', 
    'Time_With_Curr_Empr', 'pct_currentBal_all_TL', 'time_since_recent_enq', 
    'recent_level_of_deliq', 'last_prod_enq2'
]

# Create form
with st.form(key='prediction_form'):
    # Input features
    feature1 = st.number_input("Max Recent Level of Delinquency", min_value=0.0, max_value=10.0)
    feature2 = st.number_input("Number of Std 12 Mts", min_value=0, max_value=100)
    feature3 = st.number_input("Time Since Recent Payment", min_value=0, max_value=1000)
    feature4 = st.number_input("Enquiries in Last 3 Months", min_value=0, max_value=100)
    feature5 = st.number_input("PL Enquiries in Last 12 Months", min_value=0, max_value=100)
    feature6 = st.number_input("Percentage of Active TLs Ever", min_value=0.0, max_value=100.0)
    feature7 = st.number_input("Percentage of PL Enquiries in Last 6 Months of Ever", min_value=0.0, max_value=100.0)
    feature8 = st.number_input("Time with Current Employer (in years)", min_value=0, max_value=40)
    feature9 = st.number_input("Percentage of Current Balance All TL", min_value=0.0, max_value=100.0)
    feature10 = st.number_input("Time Since Recent Enquiry", min_value=0, max_value=1000)
    feature11 = st.number_input("Recent Level of Delinquency", min_value=0.0, max_value=10.0)
    categories = ['others', 'ConsumerLoan', 'PL', 'CC', 'AL', 'HL']
    # Input for Last Product Enquiry
    feature12 = st.selectbox("Last Product Enquiry", categories)

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    # Create input data with the correct types
    input_data = pd.DataFrame([{
        'max_recent_level_of_deliq': int(feature1),
        'num_std_12mts': int(feature2),
        'time_since_recent_payment': float(feature3),
        'enq_L3m': float(feature4),
        'PL_enq_L12m': float(feature5),
        'pct_of_active_TLs_ever': float(feature6),
        'pct_PL_enq_L6m_of_ever': float(feature7),
        'Time_With_Curr_Empr': int(feature8),
        'pct_currentBal_all_TL': float(feature9),
        'time_since_recent_enq': float(feature10),
        'recent_level_of_deliq': int(feature11),
        'last_prod_enq2': str(feature12)  # Convert this to string
    }])

    # Make prediction
    encoded_prediction = model.predict(input_data)[0]
    
    # Decode the label to the original class
    original_class = label_encoder.inverse_transform([encoded_prediction])[0]

    # Display the prediction
    st.write(f"Prediction: {original_class}")
