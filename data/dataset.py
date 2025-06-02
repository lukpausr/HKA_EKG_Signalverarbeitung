# dataset.py

# required imports
import os
import torch
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# from parameters import Param TODO: Delete this import if not needed in the future

# Custom Dataset for Pytorch
# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class ECG_DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, label_cols: str = None, data_cols: str = ['raw_data']):
        self.data_dir = data_dir
        self.label_cols = label_cols
        self.data_cols = data_cols
        # self.data = []

        # Generate a list containing all file names in directory
        self.file_list = os.listdir(data_dir)

        # # Read all files in directory and store them in a list
        # for file in self.file_list:
        #     # Read data from csv file
        #     temp_data = pd.read_csv(data_dir + file)
        #     # add gaussian distribution over peaks with width of 10
        #     # reason: the loss function can handle the peaks better when they have a larger range / area for the loss function to work with
        #     # for feature in Param.feature_list: TODO: Delete this comment if not needed in the future
        #     for feature in self.label_cols:
        #         if(feature == 'P-peak' or feature == 'R-peak' or feature == 'T-peak'):                   
        #             # add gaussian distribution over peaks with width of 10 // use constant to extend data by 0s when filtering with guassian
        #             temp_data[feature] = gaussian_filter1d(np.float64(temp_data[feature]), sigma=10, mode='constant')
        #             # normalize between 0 and 1
        #             max_val = max(temp_data[feature])
        #             if(max_val > 0):
        #                 temp_data[feature] = temp_data[feature] * (1/max_val)
        #             # Print Data with matplotlib
        #             #import matplotlib.pyplot as plt
        #             #plt.plot(temp_data[feature])
        #             #plt.show()          
        #     # add data to list
        #     self.data.append(temp_data)
        #     # # Print the amount of loaded files every 1000 files for better overview during loading
        #     # if len(self.data) % 1000 == 0:
        #     #     print(f"DATASET: Loaded {len(self.data)} of {len(self.file_list)} files")

    # Return the length of the dataset
    def __len__(self):
        return len(self.file_list)
    
    # Return a single data entry using a given index
    def __getitem__(self, idx):
        
        sample = pd.read_csv(self.data_dir + self.file_list[idx])

        # for feature in self.label_cols:
        #     if(feature == 'P-peak' or feature == 'R-peak' or feature == 'T-peak'):
        #         # add gaussian distribution over peaks with width of 10 // use constant to extend data by 0s when filtering with guassian
        #         sample[feature] = gaussian_filter1d(np.float64(sample[feature]), sigma=10, mode='constant')
        #         max_val = max(sample[feature])  # normalize between 0 and 1
        #         if(max_val > 0):
        #             sample[feature] = sample[feature] * (1/max_val)

        peak_features = [f for f in self.label_cols if f in ['P-peak', 'R-peak', 'T-peak']]
        if peak_features:
            arr = sample[peak_features].astype(np.float64).values
            arr = gaussian_filter1d(arr, sigma=10, axis=0, mode='constant')
            max_vals = np.max(arr, axis=0)
            # Avoid division by zero
            max_vals[max_vals == 0] = 1
            arr = arr / max_vals
            sample[peak_features] = arr

        return torch.tensor(sample[self.data_cols].values).T, torch.tensor(sample[self.label_cols].values).T

        # data_idx = self.data[idx]
        # return torch.tensor(data_idx[self.data_cols].values).T, torch.tensor(data_idx[self.label_cols].values).T
    
        # raw_data = torch.tensor(data_idx[self.data_cols].values).T
        # labels = torch.tensor(data_idx[self.label_cols].values).T
        # return raw_data, labels