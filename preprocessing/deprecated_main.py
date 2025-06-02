# main.py

# required imports
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Set the base path
# the data will be stored in this folder at /data/pd_dataset_train, /data/pd_dataset_test and /data/pd_dataset_val
base_path = r"C:\\Users\\Büro\Documents\\Projekt_Lukas\\"

# Set the path to the matlab labels
# the matlab labels are stored in this folder and will be used read-only
matlab_labels = r"C:\Users\Büro\Documents\Projekt_Lukas\Matlab_Labels\records500"

# Set the feature list as defined by the matlab labels
feature_list = ['P-wave', 'P-peak', 'QRS-comples', 'R-peak', 'T-wave', 'T-peak']

# Set the amount of augmentations per data set
augmentations = 5

# Normalization method "z-score" or "min-max"
normalization_method = "min-max"



########################################################################################
# No changes required after this line

# Start the timer for time measurement
start_time = time.time()

# Set seed for reproducibility
np.random.seed(0) # Set seed to 0 (constant) for reproducibility

# Data containers
data_with_features_train = []
data_with_features_test = []
data_with_features_validation = []

# Load all files in the directory
files_str = os.listdir(matlab_labels)
array_length = len(files_str)

# Read all files and store in a list
features_by_ecg_id = []
for i in range(0, array_length):
    features_by_ecg_id.append(pd.read_csv(matlab_labels + "/" + files_str[i]))

    # Print the amount of loaded files every 100 files for better overview during loading
    if (i % 100 == 0):
        print("Preloaded " + str(i) + " of " + str(array_length) + " samples")

for i in range(0, array_length):

    # Load raw data
    #raw_data_row_i = features_by_ecg_id[i]
    #raw_data_row_i = pd.DataFrame(raw_data_row_i)
    raw_data_row_i = pd.DataFrame(features_by_ecg_id[i])

    # Calculate the median lead of 12-lead-ecg
    median_lead = raw_data_row_i['raw_data']

    if normalization_method == "z-score":
        # Normalize median lead using Z-score normalization
        median_lead = (median_lead - np.mean(median_lead)) / np.std(median_lead)
    if normalization_method == "min-max":
        # Normalize median lead using Min-Max normalization
        median_lead = (median_lead - np.min(median_lead)) / (np.max(median_lead) - np.min(median_lead))

    # Plot the raw data and the median data
    #plt.figure(figsize=(12, 6))
    #plt.plot(raw_data_row_i['raw_data'], label='Raw Data')
    #plt.plot(median_lead, label='Median Lead', linestyle='--')
    #plt.legend()
    #plt.title(f'ECG Data for Sample {i}')
    #plt.xlabel('Data Points')
    #plt.ylabel('Amplitude')
    #plt.show()
    
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

    # Print the amount of loaded files every 100 files for better overview during loading
    if (i % 100 == 0):
        print("Processed " + str(i) + " of " + str(array_length) + " samples")

print("Augmenting and saving data...")

# Delete folders and files
shutil.rmtree(base_path + "/data/pd_dataset_train", ignore_errors=True)
shutil.rmtree(base_path + "/data/pd_dataset_test", ignore_errors=True)
shutil.rmtree(base_path + "/data/pd_dataset_val", ignore_errors=True)
# Generate Same structure again
os.makedirs(base_path + "/data/pd_dataset_train")
os.makedirs(base_path + "/data/pd_dataset_test")
os.makedirs(base_path + "/data/pd_dataset_val")

# Iterate through all elements in data_with_features_train
# For each element, save 5 (parameter: augmentations) datasets with 512 datapoints
# start at datapoint 512 and end at len(data_with_features_train[i]) - 512, select the starting point randomly
for i in range(0, len(data_with_features_train)):
    for j in range(0, augmentations):
        # Select a random starting point
        start_idx = np.random.randint(512, len(data_with_features_train[i]) - 512)
        # Extract 512 datapoints
        pd_dataset_train = data_with_features_train[i].iloc[start_idx:start_idx + 512]
        # Save the data to a file
        pd_dataset_train.to_csv(base_path + "/data/pd_dataset_train/" + str(i) + "_" + str(j) + ".csv", index=False)

# Iterate through all elements in data_with_features_test
# For each element, save 5 (parameter: augmentations) datasets with 512 datapoints
# start at datapoint 512 and end at len(data_with_features_test[i]) - 512, select the starting point randomly
for i in range(0, len(data_with_features_test)):
    for j in range(0, augmentations):
        # Select a random starting point
        start_idx = np.random.randint(512, len(data_with_features_test[i]) - 512)
        # Extract 512 datapoints
        pd_dataset_test = data_with_features_test[i].iloc[start_idx:start_idx + 512]
        # Save the data to a file
        pd_dataset_test.to_csv(base_path + "/data/pd_dataset_test/" + str(i) + "_" + str(j) + ".csv", index=False)

# Iterate through all elements in data_with_features_validation
# For each element, save 5 (parameter: augmentations) datasets with 512 datapoints
# start at datapoint 512 and end at len(data_with_features_validation[i]) - 512, select the starting point randomly
for i in range(0, len(data_with_features_validation)):
    for j in range(0, augmentations):
        # Select a random starting point
        start_idx = np.random.randint(512, len(data_with_features_validation[i]) - 512)
        # Extract 512 datapoints
        pd_dataset_validation = data_with_features_validation[i].iloc[start_idx:start_idx + 512]
        # Save the data to a file
        pd_dataset_validation.to_csv(base_path + "/data/pd_dataset_val/" + str(i) + "_" + str(j) + ".csv", index=False)

end_time = time.time()

print("Data Preprocessing took " + str(int(end_time - start_time)) + " seconds")
print("Data Preprocessing finished!")