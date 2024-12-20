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

# TODO: Tensoren in liste speichern und dann auf Liste zugreifen
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

        self.save_hyperparameters()
    
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

        loss = nn.BCEWithLogitsLoss()(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.BCEWithLogitsLoss()(x_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = nn.BCEWithLogitsLoss()(x_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

####################################################################################################

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
# From Source https://www.kaggle.com/code/super13579/u-net-1d-cnn-with-pytorch we build a U-Net Implementation using
# pytorch lightning

# Convolution + BatchNorm + ReLU Block
# QUESTION: Why is the padding set to 3? Can it be adapted to the kernel size automatically?
class conbr_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conbr_block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=3, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
    
class UNET_1D(pl.LightningModule):
    def __init__(self, in_channels, layer_n, out_channels=1, kernel_size=7, depth=3):
        super(UNET_1D, self).__init__()

        self.save_hyperparameters()

        self.example_input_array = torch.rand(1, in_channels, layer_n)

        self.in_channels = in_channels
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.out_channels = out_channels

        # Define pooling operations on encoder side
        self.AvgPool1D1 = nn.AvgPool1d(in_channels, stride=2)
        self.AvgPool1D2 = nn.AvgPool1d(in_channels, stride=2)
        self.AvgPool1D3 = nn.AvgPool1d(in_channels, stride=2)
        self.AvgPool1D4 = nn.AvgPool1d(in_channels, stride=2)

        # Apply 2 1d-convolutional layers
        # Input data size: 1 x 512
        # Output data size: 64 x 512
        self.layer1 = nn.Sequential(
            conbr_block(self.in_channels, self.in_channels * 64, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 64, self.in_channels * 64, self.kernel_size, stride=1),
        )

        # Apply 2 1d-convolutional layers
        # Input data size: 64 x 256
        # Output data size: 128 x 256
        self.layer2 = nn.Sequential(
            conbr_block(self.in_channels * 64, self.in_channels * 128, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 128, self.in_channels * 128, self.kernel_size, stride=1),
        )

        # Apply 2 1d-convolutional layers
        # Input data size: 128 x 128
        # Output data size: 256 x 128
        self.layer3 = nn.Sequential(
            conbr_block(self.in_channels * 128, self.in_channels * 256, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 256, self.in_channels * 256, self.kernel_size, stride=1),
        )
        
        # Apply 2 1d-convolutional layers
        # Input data size: 256 x 64
        # Output data size: 512 x 64
        self.layer4 = nn.Sequential(
            conbr_block(self.in_channels * 256, self.in_channels * 512, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 512, self.in_channels * 512, self.kernel_size, stride=1),
        )

        # Apply 2 1d-convolutional layers
        # Input data size: 512 x 32
        # Output data size: 1024 x 32
        self.layer5 = nn.Sequential(
            conbr_block(self.in_channels * 512, self.in_channels * 1024, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 1024, self.in_channels * 1024, self.kernel_size, stride=1),
        )

        # Transposed convolutional layers
        # Input data size: 1024 x 32
        # Output data size: 512 x 64
        self.layer5T = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels * 1024, self.in_channels * 512, self.kernel_size, stride=2, padding=3, output_padding=1),
        )

        # Double Convolutional layer and transposed convolutional layer
        # Input data size: 1024 x 64
        # Output data size: 256 x 128
        self.layer4T = nn.Sequential(
            conbr_block(self.in_channels * 1024, self.in_channels * 512, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 512, self.in_channels * 256, self.kernel_size, stride=1),
            nn.ConvTranspose1d(self.in_channels * 256, self.in_channels * 256, self.kernel_size, stride=2, padding=3, output_padding=1),
        )

        # Double Convolutional layer and transposed convolutional layer
        # Input data size: 512 x 128
        # Output data size: 128 x 256
        self.layer3T = nn.Sequential(
            conbr_block(self.in_channels * 512, self.in_channels * 256, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 256, self.in_channels * 128, self.kernel_size, stride=1),
            nn.ConvTranspose1d(self.in_channels * 128, self.in_channels * 128, self.kernel_size, stride=2, padding=3, output_padding=1),
        )

        # Double Convolutional layer and transposed convolutional layer
        # Input data size: 256 x 256
        # Output data size: 64 x 512
        self.layer2T = nn.Sequential(
            conbr_block(self.in_channels * 256, self.in_channels * 128, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 128, self.in_channels * 64, self.kernel_size, stride=1),
            nn.ConvTranspose1d(self.in_channels * 64, self.in_channels * 64, self.kernel_size, stride=2, padding=3, output_padding=1),
        )

        # Double Convolutional layer to output dimension
        # Input data size: 128 x 512
        # Output data size: out_channels x 512
        self.layer1Out = nn.Sequential(
            conbr_block(self.in_channels * 128, self.in_channels * 64, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 64, self.in_channels, self.kernel_size, stride=1),
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=1, padding=3),
        )
    
    def forward(self, x):

        enablePrint = False
        if enablePrint: print(x.size())

        #############Encoder#####################

        if enablePrint: print("Encoder Sizes")
        
        out_0 = self.layer1(x)
        if enablePrint: print(out_0.size())

        pool_out_0 = self.AvgPool1D1(out_0)
        if enablePrint: print(pool_out_0.size())

        out_1 = self.layer2(pool_out_0)
        if enablePrint: print(out_1.size())

        pool_out_1 = self.AvgPool1D2(out_1)
        if enablePrint: print(pool_out_1.size())

        out_2 = self.layer3(pool_out_1)
        if enablePrint: print(out_2.size())

        pool_out_2 = self.AvgPool1D3(out_2)
        if enablePrint: print(pool_out_2.size())

        out_3 = self.layer4(pool_out_2)
        if enablePrint: print(out_3.size())

        pool_out_3 = self.AvgPool1D4(out_3)
        if enablePrint: print(pool_out_3.size())

        out_4 = self.layer5(pool_out_3)
        if enablePrint: print(out_4.size())

        #############Decoder####################

        in_4_a = self.layer5T(out_4)
        if enablePrint: print(in_4_a.size())

        in_4 = torch.cat([in_4_a, out_3], 1)
        if enablePrint: print(in_4.size())

        in_3_a = self.layer4T(in_4)
        if enablePrint: print(in_3_a.size())

        in_3 = torch.cat([in_3_a, out_2], 1)
        if enablePrint: print(in_3.size())

        in_2_a = self.layer3T(in_3)
        if enablePrint: print(in_2_a.size())

        in_2 = torch.cat([in_2_a, out_1], 1)
        if enablePrint: print(in_2.size())

        in_1_a = self.layer2T(in_2)
        if enablePrint: print(in_1_a.size())

        in_1 = torch.cat([in_1_a, out_0], 1)
        if enablePrint: print(in_1.size())

        in_0 = self.layer1Out(in_1)
        if enablePrint: print(in_0.size())

        #out = self.outcov(up)

        x_hat = nn.functional.sigmoid(in_0)

        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = nn.BCEWithLogitsLoss()(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = nn.BCEWithLogitsLoss()(x_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = nn.BCEWithLogitsLoss()(x_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_epoch_end(self):
        print('Test Epoch End')
        print('-----------------------------------')

####################################################################################################
####################################################################################################
####################################################################################################

# Main

if __name__ == '__main__':

    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    pl.seed_everything(42)

    data_directory = r"D:\SynologyDrive\10_Arbeit_und_Bildung\20_Masterstudium\01_Semester\90_Projekt\10_DEV\data"
    enable_print = False
    batch_size = 32
    max_epochs = 5

    if enable_print:
        test_dataset = ECG_DataSet(data_dir=data_directory+'\\pd_dataset_train\\')
        x, y = test_dataset.__getitem__(0)
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)

    # Define Data Module containing train, test and validation datasets
    dm = ECG_DataModule(data_dir=data_directory, batch_size=batch_size)

    #model = ECG_Dilineation_EncDec(in_channels=1, base_channel_size=8, kernel_size=3, stride=2, padding=1, feature_channel_size=1)
    #print(ModelSummary(model, max_depth=-1))

    model = UNET_1D(in_channels=1, layer_n=512, out_channels=1, kernel_size=7, depth=2)
    print(ModelSummary(model, max_depth=-1))

    trainer = Trainer(max_epochs=max_epochs, default_root_dir=data_directory)
    trainer.fit(model=model, datamodule=dm)


# used source:
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html#Building-the-autoencoder
# https://pytorch-lightning.readthedocs.io/en/1.1.8/introduction_guide.html
# https://pytorch-lightning.readthedocs.io/en/0.10.0/introduction_guide.html#the-model
