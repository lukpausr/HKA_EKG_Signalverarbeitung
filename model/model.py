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

# From Source https://www.kaggle.com/code/super13579/u-net-1d-cnn-with-pytorch we build a U-Net Implementation using
# pytorch lightning
class conbr_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=3, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class se_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(se_block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//8, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(out_channels//8, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = nn.functional.adaptive_avg_pool1d(x, 1)
        z = self.net(y)
        x_hat = torch.add(x, z)

        return x_hat
    
class re_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(re_block, self).__init__()

        self.net = nn.Sequential(
            conbr_block(in_channels, out_channels, kernel_size, 1, dilation),
            conbr_block(out_channels, out_channels, kernel_size, 1, dilation),
            se_block(out_channels, out_channels),
        )

    def forward(self, x):
        y = self.net(x)
        x_hat = torch.add(x, y)

        return x_hat

class UNET_1D(pl.LightningModule):
    def __init__(self, in_channels, layer_n, out_channels=1, kernel_size=7, depth=3):
        super(UNET_1D, self).__init__()

        self.example_input_array = torch.rand(1, in_channels, layer_n)

        self.in_channels = in_channels
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.out_channels = out_channels

        self.AvgPool1D1 = nn.AvgPool1d(in_channels, stride=4)
        self.AvgPool1D2 = nn.AvgPool1d(in_channels, stride=16)
        self.AvgPool1D3 = nn.AvgPool1d(in_channels, stride=16*16)
        
        self.layer1 = self.down_layer(self.in_channels, self.layer_n, self.kernel_size, 1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size, 4, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.in_channels), int(self.layer_n*3), self.kernel_size, 4, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.in_channels), int(self.layer_n*4), self.kernel_size, 4, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.in_channels), int(self.layer_n*5), self.kernel_size, 4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)

        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')
        
        self.outcov = nn.Conv1d(self.layer_n, self.out_channels, kernel_size=self.kernel_size, stride=1,padding = 3)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))

        return nn.Sequential(*block)
            
    def forward(self, x):

        enablePrint = False

        if enablePrint:
            print("Input value size")
            print(x.size())
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)

        if enablePrint:
            print("Pool Sizes")
            print(pool_x1.size())
            print(pool_x2.size())
            print(pool_x3.size())
        
        #############Encoder#####################

        if enablePrint:
            print("Encoder Sizes")
        
        out_0 = self.layer1(x)

        if enablePrint:
            print("\t Out 0 / Layer 1")
            print(out_0.size())

        out_1 = self.layer2(out_0)

        if enablePrint:
            print("\t Out 1 / Layer 2")
            print(out_1.size())
        
        x = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x)

        if enablePrint:
            print("\t Out 2 / Layer 3")
            print(out_2.size())
        
        x = torch.cat([out_2, pool_x2], 1)
        x = self.layer4(x)

        if enablePrint:
            print("\t Out 3 / Layer 4")
            print(x.size())
        
        #############Decoder####################

        if enablePrint:
            print("Decoder Sizes")
        
        up = self.upsample1(x)

        if enablePrint:
            print("\t Upsample 1")
            print(up.size())

        up = torch.cat([up, out_2], 1)

        if enablePrint:
            print("\t Concat 1")
            print(up.size())

        up = self.cbr_up1(up)

        if enablePrint:
            print("\t CBR 1")
            print(up.size())
        
        up = self.upsample(up)

        if enablePrint:
            print("\t Upsample 2")
            print(up.size())

        up = torch.cat([up, out_1], 1)

        if enablePrint:
            print("\t Concat 2")
            print(up.size())

        up = self.cbr_up2(up)

        if enablePrint:
            print("\t CBR 2")
            print(up.size())
        
        up = self.upsample(up)

        if enablePrint:
            print("\t Upsample 3")
            print(up.size())

        up = torch.cat([up, out_0], 1)

        if enablePrint:
            print("\t Concat 3")
            print(up.size())

        up = self.cbr_up3(up)

        if enablePrint:
            print("\t CBR 3")
            print(up.size())
        
        out = self.outcov(up)

        if enablePrint:
            print("\t Out")
            print(out.size())

        x_hat = nn.functional.sigmoid(out)

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



    #model = ECG_Dilineation_EncDec(in_channels=1, base_channel_size=8, kernel_size=3, stride=2, padding=1, feature_channel_size=1)
    #print(ModelSummary(model, max_depth=-1))

    model2 = UNET_1D(in_channels=1, layer_n=512, out_channels=1, kernel_size=7, depth=2)
    print(ModelSummary(model2, max_depth=-1))

    trainer = Trainer(max_epochs=10)

    
    #trainer.fit(model=model2, datamodule=dm)

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