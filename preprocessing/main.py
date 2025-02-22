# main.py

# required imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import ast
# import time
import wfdb.processing
import wfdb.processing.evaluate
import wfdb.processing.qrs


# Method defined by physionet to load data
def load_raw_data(df, sampling_rate, path):
    # Loading all data with signal and meta information
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    
    # Eliminating meta information. We are selecting only signal value of 12 leads 
    data = np.array([signal for signal, meta in data])
    return data


# base_path = r"D:\SynologyDrive\10_Arbeit_und_Bildung\20_Masterstudium\01_Semester\90_Projekt\10_DEV"
base_path = r"C:\\Users\\Büro\Documents\\Projekt_Lukas\\"
path = base_path + "/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"

matlab_labels = r"C:\Users\Büro\Documents\Projekt_Lukas\Matlab_Labels\records500"


data_with_features_train = []
data_with_features_test = []
data_with_features_validation = []

save_all = False
use_matlab_data = True # always true

# Set seed for reproducibility
np.random.seed(0) #time.time_ns()%10000) # Set seed for reproducibility

# Deprecated
if use_matlab_data is False:
    features_by_ecg_id = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    features_by_ecg_id.scp_codes = features_by_ecg_id.scp_codes.apply(lambda x: ast.literal_eval(x))
    array_length = len(features_by_ecg_id)

# Currently used
if use_matlab_data is True:
    files_str = os.listdir(matlab_labels)
    array_length = len(files_str)

    # Read all files and store in a list
    features_by_ecg_id = []
    for i in range(0, array_length):
        features_by_ecg_id.append(pd.read_csv(matlab_labels + "/" + files_str[i]))

        if (i % 100 == 0):
            print("Preloaded " + str(i) + " of " + str(array_length) + " samples")

for i in range(0, array_length):
    
    # Deprecated
    if use_matlab_data is False:

        # Load raw data
        features_by_ecg_id_selected = features_by_ecg_id.iloc[i:i+1]
        raw_data_row_i = load_raw_data(features_by_ecg_id_selected, 500, path)[0]
        
        # Calculate the median lead of 12-lead-ecg
        median_lead = np.transpose(np.median(np.transpose(raw_data_row_i), axis=0))
        # Normalize median lead
        median_lead = (median_lead - np.mean(median_lead)) / np.std(median_lead)

        # Calculate the R-peaks
        rpeaks = wfdb.processing.xqrs_detect(median_lead, fs=500, verbose=False)

        # Generate feature vector and fill it with zeros, then fill it with 1 at the R-peak positions
        feature_rpeak = np.zeros(len(median_lead))
        feature_rpeak[rpeaks.astype(int)] = 1

        # Generate time id (0-4999) for each sample
        time_idx = np.arange(0, len(median_lead))
        # Convert data type of time_idx to int
        time_idx = time_idx.astype(int)
        
        # Build Pandas DataFrame containing raw data and features
        df = pd.DataFrame({'time_idx': time_idx, 'raw_data': median_lead, 'feature_rpeak': feature_rpeak})

    if use_matlab_data is True:

        # Load raw data
        #raw_data_row_i = features_by_ecg_id[i]
        #raw_data_row_i = pd.DataFrame(raw_data_row_i)
        raw_data_row_i = pd.DataFrame(features_by_ecg_id[i])

        # Features
        feature_list = ['P-wave', 'P-peak', 'QRS-comples', 'R-peak', 'T-wave', 'T-peak']
        
        # Calculate the median lead of 12-lead-ecg
        median_lead = raw_data_row_i['raw_data']
        # Normalize median lead
        median_lead = (median_lead - np.mean(median_lead)) / np.std(median_lead)
        
        df = raw_data_row_i
        # Replace 'raw-data' with median
        df['raw_data'] = median_lead

    # Use random number to define if the data is used for training or testing or validation
    # 70% of the data is used for training, 15% for testing and 15% for validation
    random_number = np.random.rand()
    if random_number < 0.7:
        data_with_features_train.append(df)
    elif random_number >= 0.7 and random_number < 0.85:
        data_with_features_test.append(df)
    else:
        data_with_features_validation.append(df)

    if (i % 100 == 0):
        print("Processed " + str(i) + " of " + str(array_length) + " samples")

if not save_all:
    # Delete folders and files
    import shutil
    shutil.rmtree(base_path + "/data/pd_dataset_train", ignore_errors=True)
    shutil.rmtree(base_path + "/data/pd_dataset_test", ignore_errors=True)
    shutil.rmtree(base_path + "/data/pd_dataset_val", ignore_errors=True)
    # Generate Same structure again
    import os
    os.makedirs(base_path + "/data/pd_dataset_train")
    os.makedirs(base_path + "/data/pd_dataset_test")
    os.makedirs(base_path + "/data/pd_dataset_val")

    # Iterate through all elements in data_with_features_train
    # For each element, save 5 datasets with 512 datapoints
    # start at datapoint 512 and end at len(data_with_features_train[i]) - 512, select the starting point randomly
    for i in range(0, len(data_with_features_train)):
        for j in range(0, 5):
            # Select a random starting point
            start_idx = np.random.randint(512, len(data_with_features_train[i]) - 512)
            # Extract 512 datapoints
            pd_dataset_train = data_with_features_train[i].iloc[start_idx:start_idx + 512]

            # Save the data to a file
            pd_dataset_train.to_csv(base_path + "/data/pd_dataset_train/" + str(i) + "_" + str(j) + ".csv", index=False)

    # Iterate through all elements in data_with_features_test
    # For each element, save 5 datasets with 512 datapoints
    # start at datapoint 512 and end at len(data_with_features_test[i]) - 512, select the starting point randomly
    for i in range(0, len(data_with_features_test)):
        for j in range(0, 5):
            # Select a random starting point
            start_idx = np.random.randint(512, len(data_with_features_test[i]) - 512)
            # Extract 512 datapoints
            pd_dataset_test = data_with_features_test[i].iloc[start_idx:start_idx + 512]

            # Save the data to a file
            pd_dataset_test.to_csv(base_path + "/data/pd_dataset_test/" + str(i) + "_" + str(j) + ".csv", index=False)

    # Iterate through all elements in data_with_features_validation
    # For each element, save 5 datasets with 512 datapoints
    # start at datapoint 512 and end at len(data_with_features_validation[i]) - 512, select the starting point randomly
    for i in range(0, len(data_with_features_validation)):
        for j in range(0, 5):
            # Select a random starting point
            start_idx = np.random.randint(512, len(data_with_features_validation[i]) - 512)
            # Extract 512 datapoints
            pd_dataset_validation = data_with_features_validation[i].iloc[start_idx:start_idx + 512]

            # Save the data to a file
            pd_dataset_validation.to_csv(base_path + "/data/pd_dataset_val/" + str(i) + "_" + str(j) + ".csv", index=False)

if save_all:
    # Convert List to DataFrame but segment it by using a column called group_id
    pd_dataset_train = pd.concat(data_with_features_train)
    pd_dataset_test = pd.concat(data_with_features_test)
    pd_dataset_validation = pd.concat(data_with_features_validation)

    # Delete variables that are not needed anymore
    del data_with_features_train
    del data_with_features_test
    del data_with_features_validation

    # Add a column to the DataFrame that segments the data into groups of 5000 samples
    pd_dataset_train['group_ids'] = np.repeat(np.arange(0, len(pd_dataset_train)/5000), 5000)
    pd_dataset_test['group_ids'] = np.repeat(np.arange(0, len(pd_dataset_test)/5000), 5000)
    pd_dataset_validation['group_ids'] = np.repeat(np.arange(0, len(pd_dataset_validation)/5000), 5000)

    # Rearrange index
    pd_dataset_train.reset_index(drop=True, inplace=True)
    pd_dataset_test.reset_index(drop=True, inplace=True)
    pd_dataset_validation.reset_index(drop=True, inplace=True)

    # Save the data to a file
    pd_dataset_train.to_csv(base_path + "/data/pd_dataset_train.csv", index=False)
    pd_dataset_test.to_csv(base_path + "/data/pd_dataset_test.csv", index=False)
    pd_dataset_validation.to_csv(base_path + "/data/pd_dataset_validation.csv", index=False)