import streamlit as st
import pandas as pd
import mlflow
import dagshub
import logging

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

# Load the dataset
df = pd.read_csv('data/processed/train.csv')

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='Kushagra-Bisht', repo_name='Credit_Risk_Modelling_MLOPs', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Kushagra-Bisht/Credit_Risk_Modelling_MLOPs.mlflow')

model_name = "model" 
model_version = 3
model_uri = f'models:/{model_name}/{model_version}'

# Cache the model loading
@st.cache_resource()
def load_model():
    return mlflow.pyfunc.load_model(model_uri)

# Load the model
model = load_model()

# Streamlit interface
st.title("Credit Risk Modelling")
st.write("Enter the features to predict the creditworthiness of an individual")

# Selected features for prediction
selected_features = [
    'max_recent_level_of_deliq', 'num_std_12mts', 'time_since_recent_payment', 
    'enq_L3m', 'PL_enq_L12m', 'pct_of_active_TLs_ever', 'pct_PL_enq_L6m_of_ever', 
    'Time_With_Curr_Empr', 'pct_currentBal_all_TL', 'time_since_recent_enq', 
    'recent_level_of_deliq', 'last_prod_enq2', "Approved_Flag"
]

# Input features (updated to match selected_features)
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
feature12 = st.number_input("Last Product Enquiry", min_value=0, max_value=100)
feature13 = st.selectbox("Approved Flag", options=[0, 1])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'max_recent_level_of_deliq': feature1,
        'num_std_12mts': feature2,
        'time_since_recent_payment': feature3,
        'enq_L3m': feature4,
        'PL_enq_L12m': feature5,
        'pct_of_active_TLs_ever': feature6,
        'pct_PL_enq_L6m_of_ever': feature7,
        'Time_With_Curr_Empr': feature8,
        'pct_currentBal_all_TL': feature9,
        'time_since_recent_enq': feature10,
        'recent_level_of_deliq': feature11,
        'last_prod_enq2': feature12,
        'Approved_Flag': feature13
    }])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f"Prediction: {prediction[0]}")
