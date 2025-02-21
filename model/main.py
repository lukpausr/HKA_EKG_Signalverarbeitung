# main.py

# required imports
import os
import math
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.model_summary import ModelSummary
import matplotlib.pyplot as plt

from parameters import Param
from datamodule import ECG_DataModule
from model import UNET_1D

# Function to generate a plot of the ECG data and the labels
def generatePlot(x, y, x_hat, y_hat):
    # Print x data (EKG-Data) in matplotlib plot and add the labels as a colored overlay to the plot
    #print(x.shape)
    #print(y.shape)
    #print(x_hat.shape)
    #print(y_hat.shape)

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
        axs[i + 1].set_ylabel(Param.feature_list[i])
        axs[i + 1].set_yticks([0, 1])
        axs[i + 1].set_yticklabels(['0', '1'])
        axs[i + 1].set_xticks([])


    axs[-1].set_xlabel('Time')
    axs[-1].set_xticks(range(0, 551, 50))

    plt.tight_layout()
    plt.show()

# Set used PC ( Training / Inference / PELU / GRMI)
used_pc = "Training"

if __name__ == '__main__':
 
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

    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Torch Version: ', torch.__version__)
    print('Using device: ', device)
    if device.type == 'cuda':
        print('Cuda Version: ', torch.version.cuda)
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

        torch.set_float32_matmul_precision('high')
                
    if(conduct_training):

        # Define Data Module containing train, test and validation datasets
        print("Initializing Data Module...")
        dm = ECG_DataModule(data_dir=data_directory, batch_size=batch_size)

        # Initialize model
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