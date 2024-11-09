import logging
import yaml
import logging
import os
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

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

from statsmodels.stats.outliers_influence import variance_inflation_factor

def filter_columns_by_vif(df, threshold=6):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    vif_data = df[numeric_columns]
    total_columns = vif_data.shape[1]
    columns_to_be_kept = []
    column_index = 0

    for i in range(total_columns):
        vif_value = variance_inflation_factor(vif_data, column_index)
        print(column_index, '---', vif_value)
        
        if vif_value <= threshold:
            columns_to_be_kept.append(numeric_columns[i])
            column_index += 1
        else:
            vif_data = vif_data.drop([numeric_columns[i]], axis=1)

    return columns_to_be_kept

def load_params(params_path):
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameter retreived successfully")
        return params
    except Exception as e:
        logger.error('Unexpected error:',e)
        raise

def save_data(train_data,test_data,data_path):
    try:
        train_data.to_csv(os.path.join(data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(data_path,'test.csv'),index=False)
        logger.debug("Train and test data saved successfully")
    except Exception as e:
        logger.error('Unexpected error:',e)
        raise

def main():
    try:
        logger.debug("Feature  Engineering Started.")
        df_merged = pd.read_csv("C:/Users/LCM/Desktop/Credit_Risk_Modelling/data/processed/cleaned_data.csv")
        logger.debug('Data loaded properly')
        
        selected_features=['max_recent_level_of_deliq', 'num_std_12mts', 'time_since_recent_payment', 'enq_L3m', 'PL_enq_L12m', 'pct_of_active_TLs_ever', 'pct_PL_enq_L6m_of_ever', 'Time_With_Curr_Empr', 'pct_currentBal_all_TL', 'time_since_recent_enq', 'recent_level_of_deliq', 'last_prod_enq2',"Approved_Flag"]

        df_merged=df_merged.loc[:,selected_features]   
        logger.debug('df_merged updated after VIF check and feature selection')
        
        params=load_params('params.yaml')
        test_size=params['feature_selection']['test_size']

        train_data,test_data=train_test_split(df_merged,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path='data/interim')  
    
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
