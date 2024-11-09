import logging
import yaml
import logging
import os
import pandas as pd
import numpy as np

# Create a logger
logger = logging.getLogger("data_cleaning")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler('cleaning_errors.log')
file_handler.setLevel(logging.ERROR)

# Create formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Function to clean missing data
def missing_clean(df_merged):
    logger.debug("Starting missing data cleaning process.")
    for i in df_merged.select_dtypes(include=['float64', 'int64']).columns:
        name = i
        logger.debug(f"Processing column: {name}")
        df_merged[i] = df_merged[i].replace(-99999.0, np.nan)
        miss = (df_merged[i].isna().sum() / len(df_merged[i])) * 100

        if miss == 0:
            logger.info(f"{i} has no missing value")
        elif miss > 20:
            df_merged.drop(columns=i, inplace=True)
            logger.info(f"{i} dropped due to more than 20% missing values.")
        else:
            # Calculate medians first, then use them in filling NaN values
            med1 = df_merged[df_merged['Approved_Flag'] == 'P1'][i].median()
            med2 = df_merged[df_merged['Approved_Flag'] == 'P3'][i].median()
            med3 = df_merged[df_merged['Approved_Flag'] == 'P2'][i].median()

            # Fill NaN values in each group with the respective median
            df_merged.loc[df_merged['Approved_Flag'] == 'P3', i] = df_merged.loc[df_merged['Approved_Flag'] == 'P3', i].fillna(med2)
            df_merged.loc[df_merged['Approved_Flag'] == 'P2', i] = df_merged.loc[df_merged['Approved_Flag'] == 'P2', i].fillna(med3)
            df_merged.loc[df_merged['Approved_Flag'] == 'P1', i] = df_merged.loc[df_merged['Approved_Flag'] == 'P1', i].fillna(med1)
            
            logger.info(f"{i} successfully imputed")

# Function for outlier treatment
def remove_classwise_iqr_outliers(df, class_column, target_columns):
    logger.debug("Starting outlier removal process.")
    # Loop through each specified target column
    for col in target_columns:
        logger.debug(f"Processing column: {col}")
        # Loop through each unique class in the specified column (Approved_Flag)
        for cls in df[class_column].unique():
            logger.debug(f"Processing class: {cls}")
            # Subset the data for the current class
            class_data = df[df[class_column] == cls]
            
            # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the column in the current class
            Q1 = class_data[col].quantile(0.25)
            Q3 = class_data[col].quantile(0.75)
            
            # Calculate IQR
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out the outliers for the current class and column
            df = df[~((df[class_column] == cls) & ((df[col] < lower_bound) | (df[col] > upper_bound)))]
            
            logger.info(f'Outliers removed in {col} for class {cls} beyond IQR bounds.')
                
    return df

def load_data(data_path):
    try:
        logger.debug(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.debug("Data loaded successfully")
        return df
    except Exception as e:
        logger.error("Failed to load data due to: %s", e)
        raise

def save_data(df_merged, data_path):
    try:
        logger.debug(f"Saving data to: {os.path.join(data_path,'cleaned_data.csv')}")
        df_merged.to_csv(os.path.join(data_path, 'cleaned_data.csv'), index=False)
        logger.debug("Data saved successfully")
    except Exception as e:
        logger.error('Unexpected error during save: %s', e)
        raise

def main():
    try:
        logger.debug("Starting the data cleaning process.")
        df_merged = pd.read_csv("https://raw.githubusercontent.com/Kushagra-Bisht/Data_CRM/refs/heads/master/Credit_rm.csv")
        logger.debug('Data loaded properly')
        
        df_merged = df_merged.drop(df_merged.columns[0], axis=1)
        logger.debug('Dropped first column.')
        
        missing_clean(df_merged)
        logger.debug('Missing values handled successfully')
        
        target_columns = ['enq_L3m', 'time_since_recent_enq', 'num_std_12mts', 'pct_PL_enq_L6m_of_ever']
        df_merged = remove_classwise_iqr_outliers(df_merged, class_column='Approved_Flag', target_columns=target_columns)
        logger.debug('Outliers handled successfully')
        
        save_data(df_merged, data_path='data/processed')  
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
