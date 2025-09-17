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
    def __init__(self, data_dir: str, label_cols: str = None, data_cols: str = None):
        self.data_dir = data_dir
        self.label_cols = label_cols
        self.data_cols = data_cols

        self.peak_to_center = False # If True, the R-peak will be centered in the 512 datapoints window

        # Generate a list containing all file names in directory
        self.file_list = os.listdir(data_dir)

    # Return the length of the dataset
    def __len__(self):
        return len(self.file_list)
    
    # # Return a single data entry using a given index
    # def __getitem__(self, idx):
    #     sample = pd.read_csv(self.data_dir + self.file_list[idx])
    #     # Augment data by selecting 512 Datapoints randomly between given bounds
    #     start_idx = np.random.randint(256, len(sample) - (512+256))
    #     sample = sample.iloc[start_idx:start_idx + 512]
    #     # Apply gaussian filter to improve learning capabilities
    #     peak_features = [f for f in self.label_cols if f in ['P-peak', 'R-peak', 'T-peak']]
    #     if peak_features:
    #         arr = sample[peak_features].astype(np.float64).values
    #         arr = gaussian_filter1d(arr, sigma=10, axis=0, mode='constant')
    #         max_vals = np.max(arr, axis=0)
    #         # Avoid division by zero
    #         max_vals[max_vals == 0] = 1
    #         arr = arr / max_vals
    #         sample[peak_features] = arr
    #     return torch.tensor(sample[self.data_cols].values, dtype=torch.float32).T, torch.tensor(sample[self.label_cols].values, dtype=torch.float32).T

    def __getitem__(self, idx):
        sample = pd.read_csv(self.data_dir + self.file_list[idx])
        
        if self.peak_to_center:
            # Find the index of the R-peak
            valid_r_peaks = []
            start_idx = None
            if 'R-peak' in self.label_cols:
                r_peak_indices = sample.index[sample['R-peak'] == 1].tolist()
                if r_peak_indices:
                    # Filter R-peaks that would allow a valid 512-point window
                    valid_r_peaks = [r for r in r_peak_indices if 256 <= r <= len(sample) - (512 + 256)]
                    if valid_r_peaks:
                        r_peak_idx = np.random.choice(valid_r_peaks)
                        start_idx = r_peak_idx - 256
            if valid_r_peaks == [] or 'R-peak' not in self.label_cols:
                # If no R-peak is found, fall back to random selection and print a warning
                start_idx = np.random.randint(256, len(sample) - (512 + 256))
                print(f"Warning: No valid R-peak found in {self.file_list[idx]}. Falling back to random window selection.")
        else:
            # Augment data by selecting 512 Datapoints randomly between given bounds
            start_idx = np.random.randint(256, len(sample) - (512 + 256))

        # Select the 512 datapoints
        sample = sample.iloc[start_idx:start_idx + 512].copy()  # Add .copy() to avoid view issues
            
        # Apply gaussian filter to improve learning capabilities
        peak_features = [f for f in self.label_cols if f in ['P-peak', 'R-peak', 'T-peak']]
        if peak_features:
            # Work with numpy arrays directly
            peak_data = sample[peak_features].astype(np.float64).values.copy()  # Explicit copy
            peak_data = gaussian_filter1d(peak_data, sigma=10, axis=0, mode='constant')
            
            max_vals = np.max(peak_data, axis=0)
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            peak_data = peak_data / max_vals
            
            # Update the DataFrame
            sample[peak_features] = peak_data
        
        # Convert to numpy arrays first, then to tensors with explicit copying
        data_array = sample[self.data_cols].values.astype(np.float32).copy()
        label_array = sample[self.label_cols].values.astype(np.float32).copy()
        
        # Create tensors from numpy arrays (this ensures proper memory layout)
        data_tensor = torch.from_numpy(data_array).T.contiguous()
        label_tensor = torch.from_numpy(label_array).T.contiguous()
        
        return data_tensor, label_tensor