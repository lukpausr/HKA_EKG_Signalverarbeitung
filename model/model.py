import os
import pandas as pd
import numpy as np
import math

from scipy.ndimage import gaussian_filter1d

import torch
from torch import nn
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import wandb
from pytorch_lightning.loggers import WandbLogger

# from sklearn import metrics
# from sklearn.model_selection import train_test_split

feature_list = ['P-wave', 'P-peak', 'QRS-comples', 'R-peak', 'T-wave', 'T-peak']

# Custom Dataset for Pytorch
# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
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
            
            # add gaussian distribution over peaks with width of 10
            # reason: the loss function can handle the peaks better when they have a larger range / area for the loss function to work with
            for feature in feature_list:
                if(feature == 'P-peak' or feature == 'R-peak' or feature == 'T-peak'):
                    
                    # add gaussian distribution over peaks with width of 10 // use constant to extend data by 0s when filtering with guassian
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

            # Print the amount of loaded files every 1000 files for better overview during loading
            if len(self.data) % 1000 == 0:
                print(f"DATASET: Loaded {len(self.data)} of {len(self.file_list)} files")

    # Return the length of the dataset
    def __len__(self):
        return len(self.file_list)
    
    # Return a single data entry using a given index
    def __getitem__(self, idx):
        data_idx = self.data[idx]
        return torch.tensor(data_idx[self.data_cols].values).T, torch.tensor(data_idx[self.label_cols].values).T
        # raw_data = torch.tensor(data_idx[self.data_cols].values).T
        # labels = torch.tensor(data_idx[self.label_cols].values).T
        # return raw_data, labels
    
# Custom Data Module for Pytorch Lightning
# Source: https://pytorch-lightning.readthedocs.io/en/1.1.8/introduction_guide.html#data
# This data module automatically handles the training, test and validation data and we don't have to worry
class ECG_DataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir: str, batch_size: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # not required because our data is already prepared through the custom dataset
        pass

    # Load the datasets
    def setup(self, stage=None):

        # the datasets instances are generated here, depending on the current stage (val/train/test split)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_train\\')
            self.val_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_val\\')
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = ECG_DataSet(data_dir=self.data_dir+'\\pd_dataset_test\\')
            pass

    # Define the train dataloader
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=True,
        )

    # Define the validation dataloader
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=False,
        )

    # Define the test dataloader
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=False,
        )

####################################################################################################
# Source of removed code
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html

####################################################################################################
# From Source https://www.kaggle.com/code/super13579/u-net-1d-cnn-with-pytorch we build a U-Net Implementation using
# pytorch lightning

# Convolution + BatchNorm + ReLU Block
# The order of Relu and Batchnorm is interchangeable and influences the performance and training speed
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

        # Calculate padding and convert padding to int
        self.padding = int(((self.kernel_size - 1) / 2))

        # Define pooling operations on encoder side
        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D4 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Factor which can be used to increase the amount of convolutional layers applied, should stay at 1 because larger values do not increase the performance
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

        # Debugging print statements
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

# Function to generate a plot of the ECG data and the labels
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
used_pc = "Inference"

if __name__ == '__main__':

    import os 
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Set seed for reproducibility
    pl.seed_everything(42)

    conduct_training = False
    conduct_test = False
    enable_print = False
    batch_size = 32
    max_epochs = 50

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

        # Initialize Trainer with wandb logger, using early stopping callback (https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)
        trainer = Trainer(
            max_epochs=max_epochs, 
            default_root_dir=data_directory, 
            accelerator="auto", 
            devices="auto", 
            strategy="auto",
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')], 
            logger=wandb_logger)

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
        checkpoint_path = checkpoint_path_pre + r"\20250221_01\checkpoints\epoch=17-step=25092.ckpt"
        model = UNET_1D.load_from_checkpoint(checkpoint_path)
        model.eval()

        model.to(device)

        # Get test dataloader for later iteration
        dl = dm.test_dataloader()

        # Confusing code was required to generate a plot of the ECG data and the labels...
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
        


# used sources:
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html#Building-the-autoencoder
# https://pytorch-lightning.readthedocs.io/en/1.1.8/introduction_guide.html
# https://pytorch-lightning.readthedocs.io/en/0.10.0/introduction_guide.html#the-model
