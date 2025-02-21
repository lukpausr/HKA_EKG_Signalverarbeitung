# model.py

# required imports
import torch
from torch import nn
import pytorch_lightning as pl

# from parameters import Param 

# From Source https://www.kaggle.com/code/super13579/u-net-1d-cnn-with-pytorch we build a U-Net Implementation using
# pytorch lightning
####################################################################################################
# Source of removed code
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html


# Convolution + BatchNorm + ReLU Block (conbr)
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
        self.loss = nn.BCELoss() # 20250214_05 # 20250215_01 # 20250215_02 # 20250221_01
    
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