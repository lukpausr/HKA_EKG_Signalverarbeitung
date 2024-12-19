# import pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from pytorch_lightning import Trainer


# from pytorch_forecasting import TimeSeriesDataSet


import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split

import os

class ECG_DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, label_cols: str = ['feature_rpeak'], data_cols: str = ['raw_data']):
        self.data_dir = data_dir
        self.label_cols = label_cols
        self.data_cols = data_cols

        # Generate a list containing all file names in directory
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file = pd.read_csv(self.data_dir + self.file_list[idx])
        data = torch.tensor(file[self.data_cols].values).T
        labels = torch.tensor(file[self.label_cols].values).T

        return data, labels
    
class ECG_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):

        # make assignments here (val/train/test split)
        # called on every process in DDP

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_train\\')
            self.val_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_val\\')
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_test\\')
            pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=False,
        )

####################################################################################################

class ECG_Dilineation_EncDec(pl.LightningModule):

    def __init__(
        self, 
        in_channels: int,
        base_channel_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        feature_channel_size: int,
        width: int = 512,
    ):
        super().__init__()

        self.in_channels = 1
        self.feature_channel_size = 1

        # self.save_hyperparameters()
    
        self.encoder = Encoder(in_channels, base_channel_size, kernel_size, stride, padding)
        self.decoder = Decoder(in_channels, base_channel_size, feature_channel_size, kernel_size, stride, padding)

        self.example_input_array = torch.rand(1, in_channels, width)
    
    def forward(self, x):
        # The forward function takes an 1d-Array of ECG-Sensor-Data and returns n channels of 1d-Data of the same shape
        z = self.encoder(x)
        x = self.decoder(z)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.CrossEntropyLoss()(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.CrossEntropyLoss()(x_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.CrossEntropyLoss()(x_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
class Encoder(nn.Module):
    def __init__(self, in_channels, base_channel_size: int, kernel_size=3, stride=2, padding=1):
        
        """Encoder:
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (int): Stride
            padding (int): Padding

        """        
        super(Encoder, self).__init__()

        hidden_channels = base_channel_size

        self.example_input_array = torch.rand(1, 512)

        self.net = nn.Sequential(
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),                # 512 -> 256
            nn.ReLU(),                                                                                  # ReLU activation
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),            # 256 -> 128
            nn.ReLU(),                                                                                  # ReLU activation
            nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size=3, stride=2, padding=1),        # 128 -> 64
            nn.ReLU(),                                                                                  # ReLU activation
            nn.Conv1d(2 * hidden_channels, 4 * hidden_channels, kernel_size=3, stride=2, padding=1),    # 64 -> 32
            nn.ReLU(),                                                                                  # ReLU activation
            nn.Flatten(start_dim=1, end_dim=-1),                                                        # 32 * 4 * hidden_channels -> 32 * 4 * hidden_channels
            nn.Linear(32 * hidden_channels * 4, 128)                                                    # 32 * 4 * hidden_channels -> 128
        )
        
    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(self, in_channels, base_channel_size: int, feature_channel_size: int, kernel_size=3, stride=1, padding=1):

        """Decoder:
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size
            stride (int): Stride
            padding (int): Padding

        """

        super(Decoder, self).__init__()

        self.hidden_channels = base_channel_size

        self.linear = nn.Sequential(
            nn.Linear(128, 32 * 4 * self.hidden_channels),                                                                                       # 128 -> 32 * 4 * hidden_channels
            nn.ReLU()                                                                                                                       # ReLU activation
        )

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4 * self.hidden_channels, 2 * self.hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),             # 32 -> 64
            nn.ReLU(),                                                                                                                      # ReLU activation
            nn.ConvTranspose1d(2 * self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),                 # 64 -> 128
            nn.ReLU(),                                                                                                                      # ReLU activation
            nn.ConvTranspose1d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),                     # 128 -> 256
            nn.ReLU(),                                                                                                                      # ReLU activation
            nn.ConvTranspose1d(self.hidden_channels, in_channels * feature_channel_size, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256 -> 512
            nn.Sigmoid()                                                                                                                    # Sigmoid activation
        )
        
    def forward(self, x):
        x = self.linear(x)
        print(x.shape)
        x = torch.reshape(x, (-1, 4 * self.hidden_channels, 32))
        x = self.net(x)
        return x
        
####################################################################################################
# Main

if __name__ == '__main__':

    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    pl.seed_everything(42)
    data_directory = r"D:\SynologyDrive\10_Arbeit_und_Bildung\20_Masterstudium\01_Semester\90_Projekt\10_DEV\data"


    test_dataset = ECG_DataSet(data_dir=data_directory+'\\pd_dataset_train\\')
    x, y = test_dataset.__getitem__(0)

    print(x)
    print(y)
    print(x.shape)
    print(y.shape)

    dm = ECG_DataModule(data_dir=data_directory, batch_size=8)
    
    #dm.setup()
    #dl = dm.train_dataloader()

    #dl = dm.train_dataloader()


    #print(dl)



    model = ECG_Dilineation_EncDec(in_channels=1, base_channel_size=8, kernel_size=3, stride=2, padding=1, feature_channel_size=1)
    print(ModelSummary(model, max_depth=-1))

    trainer = Trainer(max_epochs=10)
    trainer.fit(model=model, datamodule=dm)

    #trainer.test(model = model, datamodule = dm)


















# used source:
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html#Building-the-autoencoder
# https://pytorch-lightning.readthedocs.io/en/1.1.8/introduction_guide.html
# https://pytorch-lightning.readthedocs.io/en/0.10.0/introduction_guide.html#the-model



# from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

# data_dir = r"D:\SynologyDrive\10_Arbeit_und_Bildung\20_Masterstudium\01_Semester\90_Projekt\10_DEV\data"
# train_df = pd.read_csv(data_dir + "/pd_dataset_train.csv")

# context_length = 512
# prediction_length = 512
# training_cutoff = 512

# x = TimeSeriesDataSet(
#     data = train_df[lambda x: x.time_idx < training_cutoff], 
#     time_idx = "time_idx", 
#     target = "feature_rpeak", 
#     target_normalizer = GroupNormalizer(groups = ["group_ids"]),
#     categorical_encoders = {"group_ids": NaNLabelEncoder().fit(train_df.group_ids)},
#     group_ids = ["group_ids"],
#     min_encoder_length = context_length,
#     max_encoder_length = context_length,
#     min_prediction_length = prediction_length,
#     max_prediction_length = prediction_length,
