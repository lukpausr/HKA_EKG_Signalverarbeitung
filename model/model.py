import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn import metrics





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
        self.feature_channel_size = 5

        # self.save_hyperparameters()
    
        self.encoder = Encoder(in_channels, base_channel_size, kernel_size, stride, padding)
        self.decoder = Decoder(in_channels, base_channel_size, feature_channel_size, kernel_size, stride, padding)

        self.example_input_array = torch.rand(in_channels, width)
    
    def forward(self, x):
        # The forward function takes an 1d-Array of ECG-Sensor-Data and returns n channels of 1d-Data of the same shape
        z = self.encoder(x)
        x = self.decoder(z)
        return x

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
            nn.Flatten(start_dim=0, end_dim=-1),                                                        # 32 * 4 * hidden_channels -> 32 * 4 * hidden_channels
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

        hidden_channels = base_channel_size

        self.linear = nn.Sequential(
            nn.Linear(128, 32 * 4 * hidden_channels),                                                                                       # 100 -> 31 * 4 * hidden_channels
            nn.ReLU()                                                                                                                       # ReLU activation
        )

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4 * hidden_channels, 2 * hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),             # 31 -> 62
            nn.ReLU(),                                                                                                                      # ReLU activation
            nn.ConvTranspose1d(2 * hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),                 # 62 -> 125
            nn.ReLU(),                                                                                                                      # ReLU activation
            nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),                     # 125 -> 250
            nn.ReLU(),                                                                                                                      # ReLU activation
            nn.ConvTranspose1d(hidden_channels, in_channels * feature_channel_size, kernel_size=3, stride=2, padding=1, output_padding=1),  # 250 -> 500
            nn.Tanh()                                                                                                                       # Tanh activation
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (32, 32))
        x = self.net(x)
        return x
        
model = ECG_Dilineation_EncDec(in_channels=1, base_channel_size=8, kernel_size=3, stride=2, padding=1, feature_channel_size=5)

from pytorch_lightning.utilities.model_summary import ModelSummary
print(ModelSummary(model, max_depth=-1))