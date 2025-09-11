# datamodule.py

# required imports
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# from parameters import Param
from data.dataset import ECG_DataSet

# Custom Data Module for Pytorch Lightning
# Source: https://pytorch-lightning.readthedocs.io/en/1.1.8/introduction_guide.html#data
# This data module automatically handles the training, test and validation data and we don't have to worry
class ECG_DataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir, batch_size=32, num_workers=1, transform=None, persistent_workers=True, feature_list=None, data_cols=['I']):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform      # Not used right now, but prepared for future data augmentation if required
        self.persistent_workers = persistent_workers
        self.feature_list = feature_list
        self.data_cols = data_cols

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # not required because our data is already prepared through the custom dataset
        pass

    # Load the datasets
    def setup(self, stage=None):

        # Raise an error if transform != None because it hasn't been implemented yet
        if self.transform is not None:
            raise NotImplementedError("Transform is not implemented yet and does not have any influence on the data. Currently, transforms are handled inside dataset.py and are hardcoded.")

        # the datasets instances are generated here, depending on the current stage (val/train/test split)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_train\\', label_cols=self.feature_list, data_cols=self.data_cols)
            self.val_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_val\\', label_cols=self.feature_list, data_cols=self.data_cols)
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_test\\', label_cols=self.feature_list, data_cols=self.data_cols)
            pass

    # Define the train dataloader
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    # Define the validation dataloader
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    # Define the test dataloader
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )