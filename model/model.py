import os
import pandas as pd
import numpy as np
import math

from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning import Trainer

import wandb
from pytorch_lightning.loggers import WandbLogger

from sklearn import metrics
from sklearn.model_selection import train_test_split

feature_list = ['P-wave', 'P-peak', 'QRS-comples', 'R-peak', 'T-wave', 'T-peak']

class ECG_DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, label_cols: str = feature_list, data_cols: str = ['raw_data']):
        self.data_dir = data_dir
        self.label_cols = label_cols
        self.data_cols = data_cols
        self.data = []

        # Generate a list containing all file names in directory
        self.file_list = os.listdir(data_dir)
        # Read all files in directory and store them in a list
        for file in self.file_list:
            # Read data from csv file
            temp_data = pd.read_csv(data_dir + file)
            
            # add gaussian distribution over peaks with width of 5
            for feature in feature_list:
                if(feature == 'P-peak' or feature == 'R-peak' or feature == 'T-peak'):
                    
                    # add gaussian distribution over peaks with width of 5 // use constant to extend data by 0s when filtering with guassian
                    temp_data[feature] = gaussian_filter1d(np.float64(temp_data[feature]), sigma=10, mode='constant')
                    # normalize between 0 and 1
                    max_val = max(temp_data[feature])
                    if(max_val > 0):
                        temp_data[feature] = temp_data[feature] * (1/max_val)

                    # Print Data with matplotlib
                    #import matplotlib.pyplot as plt
                    #plt.plot(temp_data[feature])
                    #plt.show()
            
            # add data to list
            self.data.append(temp_data)

            if len(self.data) % 1000 == 0:
                print(f"DATASET: Loaded {len(self.data)} of {len(self.file_list)} files")

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # file = pd.read_csv(self.data_dir + self.file_list[idx])
        data_idx = self.data[idx]
        raw_data = torch.tensor(data_idx[self.data_cols].values).T
        labels = torch.tensor(data_idx[self.label_cols].values).T

        return raw_data, labels
    
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
        # print(x.shape)
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
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same', bias=True),
            # nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.net(x)
    
class UNET_1D(pl.LightningModule):
    def __init__(self, in_channels, layer_n, out_channels=1, kernel_size=3):
        super(UNET_1D, self).__init__()

        self.save_hyperparameters()

        # self.loss = nn.BCEWithLogitsLoss() # 20250214_04
        # self.loss = nn.CrossEntropyLoss() # 20250214_03
        self.loss = nn.BCELoss() # 20250214_05 # 20250215_01 # 20250215_02
    

        self.example_input_array = torch.rand(1, in_channels, layer_n)

        self.in_channels = in_channels
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        # Calculaete padding and Convert padding to int
        self.padding = int(((self.kernel_size - 1) / 2))

        # Define pooling operations on encoder side
        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D4 = nn.AvgPool1d(kernel_size=2, stride=2)

        factor = 1

        # Apply 2 1d-convolutional layers
        # Input data size: 1 x 512
        # Output data size: 64 x 512
        self.layer1 = nn.Sequential(
            conbr_block(self.in_channels, self.in_channels * 64 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 64 * factor, self.in_channels * 64 * factor, self.kernel_size, stride=1),
        )

        # Apply 2 1d-convolutional layers
        # Input data size: 64 x 256
        # Output data size: 128 x 256
        self.layer2 = nn.Sequential(
            conbr_block(self.in_channels * 64 * factor, self.in_channels * 128 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 128 * factor, self.in_channels * 128 * factor, self.kernel_size, stride=1),
        )

        # Apply 2 1d-convolutional layers
        # Input data size: 128 x 128
        # Output data size: 256 x 128
        self.layer3 = nn.Sequential(
            conbr_block(self.in_channels * 128 * factor, self.in_channels * 256 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 256 * factor, self.in_channels * 256 * factor, self.kernel_size, stride=1),
        )
        
        # Apply 2 1d-convolutional layers
        # Input data size: 256 x 64
        # Output data size: 512 x 64
        self.layer4 = nn.Sequential(
            conbr_block(self.in_channels * 256 * factor, self.in_channels * 512 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 512 * factor, self.in_channels * 512 * factor, self.kernel_size, stride=1),
        )

        # Apply 2 1d-convolutional layers
        # Input data size: 512 x 32
        # Output data size: 1024 x 32
        self.layer5 = nn.Sequential(
            conbr_block(self.in_channels * 512 * factor, self.in_channels * 1024 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 1024 * factor, self.in_channels * 1024 * factor, self.kernel_size, stride=1),
        )

        # Transposed convolutional layers
        # Input data size: 1024 x 32
        # Output data size: 512 x 64
        self.layer5T = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels * 1024 * factor, self.in_channels * 512 * factor, self.kernel_size, stride=2, padding=self.padding, output_padding=1),
        )

        # Double Convolutional layer and transposed convolutional layer
        # Input data size: 1024 x 64
        # Output data size: 256 x 128
        self.layer4T = nn.Sequential(
            conbr_block(self.in_channels * 1024 * factor, self.in_channels * 512 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 512 * factor, self.in_channels * 256 * factor, self.kernel_size, stride=1),
            nn.ConvTranspose1d(self.in_channels * 256 * factor, self.in_channels * 256 * factor, self.kernel_size, stride=2, padding=self.padding, output_padding=1),
        )

        # Double Convolutional layer and transposed convolutional layer
        # Input data size: 512 x 128
        # Output data size: 128 x 256
        self.layer3T = nn.Sequential(
            conbr_block(self.in_channels * 512 * factor, self.in_channels * 256 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 256 * factor, self.in_channels * 128 * factor, self.kernel_size, stride=1),
            nn.ConvTranspose1d(self.in_channels * 128 * factor, self.in_channels * 128 * factor, self.kernel_size, stride=2, padding=self.padding, output_padding=1),
        )

        # Double Convolutional layer and transposed convolutional layer
        # Input data size: 256 x 256
        # Output data size: 64 x 512
        self.layer2T = nn.Sequential(
            conbr_block(self.in_channels * 256 * factor, self.in_channels * 128 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 128 * factor, self.in_channels * 64 * factor, self.kernel_size, stride=1),
            nn.ConvTranspose1d(self.in_channels * 64 * factor, self.in_channels * 64 * factor, self.kernel_size, stride=2, padding=self.padding, output_padding=1),
        )

        # Double Convolutional layer to output dimension
        # Input data size: 128 x 512
        # Output data size: out_channels x 512
        self.layer1Out = nn.Sequential(
            conbr_block(self.in_channels * 128 * factor, self.in_channels * 64 * factor, self.kernel_size, stride=1),
            conbr_block(self.in_channels * 64 * factor, self.out_channels, self.kernel_size, stride=1),
            # conbr_block(self.in_channels * 64 * factor, self.in_channels, self.kernel_size, stride=1),
            # nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=1, padding='same'),
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
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = self.loss(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = self.loss(x_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        loss = self.loss(x_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_epoch_end(self):
        print('Test Epoch End')
        print('-----------------------------------')

####################################################################################################
####################################################################################################
####################################################################################################

def generatePlot(x, y, x_hat, y_hat):
    # Print x data (EKG-Data) in matplotlib plot and add the labels as a colored overlay to the plot
    import matplotlib.pyplot as plt

    print(x.shape)
    print(y.shape)
    print(x_hat.shape)
    print(y_hat.shape)

    fig, axs = plt.subplots(7, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [5, 1, 1, 1, 1, 1, 1]})

    # Plot ECG data
    axs[0].plot(x[0].numpy(), color='blue')
    axs[0].set_title('ECG Data')
    # Range between min and max value of ECG data
    axs[0].set_ylim([min(x[0].numpy())*1.1, max(x[0].numpy())*1.1])
    axs[0].set_ylabel('Amplitude')
    axs[0].set_yticks(range(math.floor(min(x[0].numpy())*1.1),math.floor(max(x[0].numpy())*1.1), 1))
    axs[0].set_xticks(range(0, 551, 50))

    # Plot labels
    for i in range(6):
        axs[i + 1].plot(y[i].numpy(), color='green')
        axs[i + 1].plot(y_hat[i].numpy(), color='blue')
        axs[i + 1].set_ylim([-0.1, 1.1])
        axs[i + 1].set_ylabel(feature_list[i])
        axs[i + 1].set_yticks([0, 1])
        axs[i + 1].set_yticklabels(['0', '1'])
        axs[i + 1].set_xticks([])


    axs[-1].set_xlabel('Time')
    axs[-1].set_xticks(range(0, 551, 50))

    plt.tight_layout()
    plt.show()



# Main
# Set used PC ( Training / Inference / PELU / GRMI)
used_pc = "Training"

if __name__ == '__main__':

    import os 
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Set seed for reproducibility
    pl.seed_everything(42)

    conduct_training = False
    conduct_test = False
    enable_print = False
    batch_size = 32
    max_epochs = 20

    if (used_pc == "Training"):
        data_directory = r"C:\Users\Büro\Documents\Projekt_Lukas\data"
        conduct_training = True
        conduct_test = False
    if (used_pc == "Inference"):
        data_directory = r"C:\Users\Büro\Documents\Projekt_Lukas\data"
        conduct_training = False
        conduct_test = True
    if (used_pc == "PELU"):
        data_directory = r"D:\SynologyDrive\10_Arbeit_und_Bildung\20_Masterstudium\01_Semester\90_Projekt\10_DEV\data"
    if (used_pc == "GRMI"):
        data_directory = r"..."  

    print(torch.__version__)
    print(torch.version.cuda)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

        torch.set_float32_matmul_precision('high')
  
    if enable_print:
        test_dataset = ECG_DataSet(data_dir=data_directory+'\\pd_dataset_train\\')
        x, y = test_dataset.__getitem__(0)
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)
   
    if(1==2):
        dl = dm.train_dataloader()
        for i in range(10):

            item = dl.dataset.__getitem__(i)

            print("Loading Batch...")

            x, y = item
            print(x.shape)
            print(y.shape)
            
            # Print x data (EKG-Data) in matplotlib plot and add the labels as a colored overlay to the plot
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(7, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [5, 1, 1, 1, 1, 1, 1]})
            
            # Plot ECG data
            axs[0].plot(x[0].numpy(), color='blue')
            axs[0].set_title('ECG Data')
            # Range between min and max value of ECG data
            axs[0].set_ylim([min(x[0].numpy())*1.1, max(x[0].numpy())*1.1])
            axs[0].set_ylabel('Amplitude')
            axs[0].set_yticks(range(math.floor(min(x[0].numpy())*1.1),math.floor(max(x[0].numpy())*1.1), 1))
            axs[0].set_xticks(range(0, 551, 50))

            # Plot labels
            for i in range(6):
                axs[i + 1].plot(y[i].numpy(), color='green')
                axs[i + 1].set_ylim([-0.1, 1.1])
                axs[i + 1].set_ylabel(feature_list[i])
                axs[i + 1].set_yticks([0, 1])
                axs[i + 1].set_yticklabels(['0', '1'])
                axs[i + 1].set_xticks([])

            axs[-1].set_xlabel('Time')
            axs[-1].set_xticks(range(0, 551, 50))

            plt.tight_layout()
            plt.show()
            
    if(conduct_training):

        # Define Data Module containing train, test and validation datasets
        print("Initializing Data Module...")
        dm = ECG_DataModule(data_dir=data_directory, batch_size=batch_size)

        model = UNET_1D(in_channels=1, layer_n=512, out_channels=6, kernel_size=5)
        print(ModelSummary(model, max_depth=-1))

        # Initialize logger on wandb
        # Source on how to setup wandb logger: https://wandb.ai/HKA-EKG-Signalverarbeitung
        wandb_logger = WandbLogger(project='HKA-EKG-Signalverarbeitung')

        # Add batch size to wandb config
        wandb_logger.experiment.config["batch_size"] = batch_size

        # Initialize Trainer with wandb logger
        trainer = Trainer(max_epochs=max_epochs, default_root_dir=data_directory, accelerator="auto", devices="auto", strategy="auto", logger=wandb_logger)

        #trainer = Trainer(max_epochs=max_epochs, default_root_dir=data_directory, logger=wandb_logger)
        trainer.fit(model=model, datamodule=dm)

        # Finish wandb
        wandb.finish()

    if(conduct_test):

        # Define Data Module containing train, test and validation datasets
        print("Initializing Data Module...")
        dm = ECG_DataModule(data_dir=data_directory, batch_size=batch_size)
        dm.setup(stage="test")

        checkpoint_path_pre = r"\\nas-k2\homes\Lukas Pelz\10_Arbeit_und_Bildung\20_Masterstudium\01_Semester\90_Projekt\10_DEV\HKA_EKG_Signalverarbeitung\HKA-EKG-Signalverarbeitung"
        checkpoint_path = checkpoint_path_pre + r"\20250216_03\checkpoints\epoch=49-step=69700.ckpt"
        model = UNET_1D.load_from_checkpoint(checkpoint_path)
        model.eval()

        model.to(device)

        dl = dm.test_dataloader()

        for i in range(100):
            x_val, y_val = dl.dataset.__getitem__(i)    
            x_val.resize_(1, 1, 512)
            y_val.resize_(1, 6, 512)

            x_val = x_val.float()
            y_val = y_val.float()

            x_val = x_val.to(device)
            y_val = y_val.to(device)

            print(x_val.shape)
            print(y_val.shape)

            y_hat = model(x_val)

            y_hat = y_hat.cpu()
            x_val = x_val.cpu()
            y_val = y_val.cpu()

            x_val = x_val[0].detach()
            y_val = y_val[0].detach()
            y_hat = y_hat[0].detach()
            generatePlot(x_val, y_val, x_val, y_hat)
        
# used source:
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html#Building-the-autoencoder
# https://pytorch-lightning.readthedocs.io/en/1.1.8/introduction_guide.html
# https://pytorch-lightning.readthedocs.io/en/0.10.0/introduction_guide.html#the-model
