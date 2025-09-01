# model.py

# required imports
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Metric
from torchmetrics.classification import BinaryJaccardIndex
from matplotlib import pyplot as plt

import numpy as np

# from parameters import Param 

# From Source https://www.kaggle.com/code/super13579/u-net-1d-cnn-with-pytorch we build a U-Net Implementation using
# pytorch lightning
####################################################################################################
# Source of removed code
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html

# Multi Tolerance Wrapper
class MultiToleranceWrapper(nn.Module):
    """
    A wrapper module for handling multiple tolerance-based metrics in parallel.
    This class manages a set of `CustomMetrics` instances, each configured with a different tolerance (in milliseconds).
    It provides unified methods to update, compute, and reset all metrics simultaneously.
    Args:
        tol_ms (list[float], optional): List of tolerance values in milliseconds. Defaults to [5, 10, 40, 150].
        sampling_rate (int, optional): The sampling rate of the signal. Defaults to 512.
    """

    def __init__(self,  tol_ms: float = [5, 10, 40, 150], sampling_rate: int = 512):
        """
        Initializes the MultiToleranceWrapper with multiple CustomMetrics, each for a specified tolerance.
        Args:
            model (nn.Module): The underlying model.
            tol_ms (list[float], optional): List of tolerance values in milliseconds.
            sampling_rate (int, optional): The sampling rate of the signal.
        """
        super().__init__()
        self.metrics = nn.ModuleDict({
            f"tolerance_{tol_m}ms": CustomMetrics(tol_ms=tol_m, sampling_rate=sampling_rate) for tol_m in tol_ms
        })

    def update(self, y_true, y_pred):
        """
        Updates all contained metrics with new predictions and ground truth values.
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
        """
        for metric in self.metrics.values():
            metric.update(y_true, y_pred)

    def compute(self):
        """
        Computes and aggregates the results from all contained metrics.
        Returns:
            dict: A dictionary containing computed metrics for each tolerance.
        """
        results = {}
        for name, metric in self.metrics.items():
            computed_metric = metric.compute()
            results.update({f"{name}_{k}": v.float() if isinstance(v, torch.Tensor) else v for k, v in computed_metric.items()})
        return results

    def reset(self):
        """
        Resets all contained metrics to their initial state.
        """
        for metric in self.metrics.values():
            metric.reset()

# Custom metrics class for evaluation
# Find help on https://pytorch-lightning.readthedocs.io/en/0.10.0/metrics.html#
class CustomMetrics(Metric):

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, tol_ms: float = 40, sampling_rate: int = 512):
        super().__init__()
        self.tol_ms = tol_ms
        self.sampling_rate = sampling_rate

        self.add_state ("tp_onset", default=torch.tensor([0 for _ in range(6)]), dist_reduce_fx=self.custom_reduction)
        self.add_state ("fp_onset", default=torch.tensor([0 for _ in range(6)]), dist_reduce_fx=self.custom_reduction)
        self.add_state ("fn_onset", default=torch.tensor([0 for _ in range(6)]), dist_reduce_fx=self.custom_reduction)
        self.add_state ("tp_offset", default=torch.tensor([0 for _ in range(6)]), dist_reduce_fx=self.custom_reduction)
        self.add_state ("fp_offset", default=torch.tensor([0 for _ in range(6)]), dist_reduce_fx=self.custom_reduction)
        self.add_state ("fn_offset", default=torch.tensor([0 for _ in range(6)]), dist_reduce_fx=self.custom_reduction)

        # regular instance variables (non state)
        for i in range(6):
            setattr(self, f"onset_differences_ms_category_{i}", [])
            setattr(self, f"offset_differences_ms_category_{i}", [])

    def custom_reduction(self, tensor):
        return torch.sum(tensor, dim=1)

    def update(self, y_true, y_pred):
        y_true_batch_cpu = y_true.cpu().numpy()
        y_pred_batch_cpu = y_pred.cpu().numpy()

        # Temporary metric state for array generation
        tp_onset = torch.tensor([ 0 for _ in range(6)])
        fp_onset = torch.tensor([ 0 for _ in range(6)])
        fn_onset = torch.tensor([ 0 for _ in range(6)])
        tp_offset = torch.tensor([ 0 for _ in range(6)])
        fp_offset = torch.tensor([ 0 for _ in range(6)])
        fn_offset = torch.tensor([ 0 for _ in range(6)])

        for y_true, y_pred in zip(y_true_batch_cpu, y_pred_batch_cpu):
            # Calculate the true and predicted onsets and offsets
            onsets_true, onsets_pred, offsets_true, offsets_pred = self.onset_offset_extraction(y_true, y_pred)

            # For each category, calculate which onsets and offsets matches together
            for category in range(6):

                matched_onsets, matched_offsets = self.greedyMatchingAlgorithm(
                    onsets_true[category].tolist(), 
                    onsets_pred[category].tolist(), 
                    offsets_true[category].tolist(), 
                    offsets_pred[category].tolist()
                    )

                onset_differences = np.array([pred - true for true, pred in matched_onsets]) if matched_onsets else np.array([]).tolist()
                offset_differences = np.array([pred - true for true, pred in matched_offsets]) if matched_offsets else np.array([]).tolist()

                onset_differences_ms = np.array([diff * (1000 / self.sampling_rate) for diff in onset_differences])
                offset_differences_ms = np.array([diff * (1000 / self.sampling_rate) for diff in offset_differences])

                # Calculate metrics TP, FP, FN and filter using given tolerance for the current category
                tp_onset[category] += np.sum((onset_differences_ms >= -self.tol_ms) & (onset_differences_ms <= self.tol_ms))
                fp_onset[category] += np.sum(onset_differences_ms > self.tol_ms)
                fn_onset[category] += np.sum(onset_differences_ms < -self.tol_ms)
                tp_offset[category] += np.sum((offset_differences_ms >= -self.tol_ms) & (offset_differences_ms <= self.tol_ms))
                fp_offset[category] += np.sum(offset_differences_ms > self.tol_ms)
                fn_offset[category] += np.sum(offset_differences_ms < -self.tol_ms)

                match category:
                    case 0:
                        self.onset_differences_ms_category_0.append(onset_differences_ms)
                        self.offset_differences_ms_category_0.append(offset_differences_ms)
                    case 1:
                        self.onset_differences_ms_category_1.append(onset_differences_ms)
                        self.offset_differences_ms_category_1.append(offset_differences_ms)
                    case 2:
                        self.onset_differences_ms_category_2.append(onset_differences_ms)
                        self.offset_differences_ms_category_2.append(offset_differences_ms)
                    case 3:
                        self.onset_differences_ms_category_3.append(onset_differences_ms)
                        self.offset_differences_ms_category_3.append(offset_differences_ms)
                    case 4:
                        self.onset_differences_ms_category_4.append(onset_differences_ms)
                        self.offset_differences_ms_category_4.append(offset_differences_ms)
                    case 5:
                        self.onset_differences_ms_category_5.append(onset_differences_ms)
                        self.offset_differences_ms_category_5.append(offset_differences_ms)

        self.tp_onset += tp_onset.detach().clone().cuda()
        self.fp_onset += fp_onset.detach().clone().cuda()
        self.fn_onset += fn_onset.detach().clone().cuda()
        self.tp_offset += tp_offset.detach().clone().cuda()
        self.fp_offset += fp_offset.detach().clone().cuda()
        self.fn_offset += fn_offset.detach().clone().cuda()

    def compute(self):

        tp_onset = torch.tensor([0 for _ in range(6)])
        fp_onset = torch.tensor([0 for _ in range(6)])
        fn_onset = torch.tensor([0 for _ in range(6)])
        tp_offset = torch.tensor([0 for _ in range(6)])
        fp_offset = torch.tensor([0 for _ in range(6)])
        fn_offset = torch.tensor([0 for _ in range(6)])
        sensitivity_onset = torch.tensor([0 for _ in range(6)])
        ppv_onset = torch.tensor([0 for _ in range(6)])
        f1_onset = torch.tensor([0 for _ in range(6)])
        sensitivity_offset = torch.tensor([0 for _ in range(6)])
        ppv_offset = torch.tensor([0 for _ in range(6)])
        f1_offset = torch.tensor([0 for _ in range(6)])

        # Compute metrics for each category individually
        for category in range(6):
            tp_onset[category], fp_onset[category], fn_onset[category] = int(self.tp_onset[category]), int(self.fp_onset[category]), int(self.fn_onset[category])
            tp_offset[category], fp_offset[category], fn_offset[category] = int(self.tp_offset[category]), int(self.fp_offset[category]), int(self.fn_offset[category])
            sensitivity_onset[category] = tp_onset[category] / (tp_onset[category] + fn_onset[category]) if (tp_onset[category] + fn_onset[category]) > 0 else 0
            sensitivity_offset[category] = tp_offset[category] / (tp_offset[category] + fn_offset[category]) if (tp_offset[category] + fn_offset[category]) > 0 else 0
            ppv_onset[category] = tp_onset[category] / (tp_onset[category] + fp_onset[category]) if (tp_onset[category] + fp_onset[category]) > 0 else 0
            ppv_offset[category] = tp_offset[category] / (tp_offset[category] + fp_offset[category]) if (tp_offset[category] + fp_offset[category]) > 0 else 0
            f1_onset[category] = 2 * (ppv_onset[category] * sensitivity_onset[category]) / (ppv_onset[category] + sensitivity_onset[category]) if (ppv_onset[category] + sensitivity_onset[category]) > 0 else 0
            f1_offset[category] = 2 * (ppv_offset[category] * sensitivity_offset[category]) / (ppv_offset[category] + sensitivity_offset[category]) if (ppv_offset[category] + sensitivity_offset[category]) > 0 else 0

        # Compute global metrics
        tp_onset_global = tp_onset.sum()
        fp_onset_global = fp_onset.sum()
        fn_onset_global = fn_onset.sum()
        tp_offset_global = tp_offset.sum()
        fp_offset_global = fp_offset.sum()
        fn_offset_global = fn_offset.sum()
        sensitivity_onset_global = tp_onset_global / (tp_onset_global + fn_onset_global) if (tp_onset_global + fn_onset_global) > 0 else 0
        sensitivity_offset_global = tp_offset_global / (tp_offset_global + fn_offset_global) if (tp_offset_global + fn_offset_global) > 0 else 0
        ppv_onset_global = tp_onset_global / (tp_onset_global + fp_onset_global) if (tp_onset_global + fp_onset_global) > 0 else 0
        ppv_offset_global = tp_offset_global / (tp_offset_global + fp_offset_global) if (tp_offset_global + fp_offset_global) > 0 else 0
        f1_onset_global = 2 * (ppv_onset_global * sensitivity_onset_global) / (ppv_onset_global + sensitivity_onset_global) if (ppv_onset_global + sensitivity_onset_global) > 0 else 0
        f1_offset_global = 2 * (ppv_offset_global * sensitivity_offset_global) / (ppv_offset_global + sensitivity_offset_global) if (ppv_offset_global + sensitivity_offset_global) > 0 else 0

        self.plot_onset_offset_histograms()

        return {
            "tp_onset_global" : tp_onset_global,
            "fp_onset_global" : fp_onset_global,
            "fn_onset_global" : fn_onset_global,
            "tp_offset_global" : tp_offset_global,
            "fp_offset_global" : fp_offset_global,
            "fn_offset_global" : fn_offset_global,
            "sensitivity_onset_global" : sensitivity_onset_global,
            "sensitivity_offset_global" : sensitivity_offset_global,
            "ppv_onset_global" : ppv_onset_global,
            "ppv_offset_global" : ppv_offset_global,
            "f1_onset_global" : f1_onset_global,
            "f1_offset_global" : f1_offset_global
            # "tp_onset": tp_onset,
            # "fp_onset": fp_onset,
            # "fn_onset": fn_onset,
            # "tp_offset": tp_offset,
            # "fp_offset": fp_offset,
            # "fn_offset": fn_offset,
            # "sensitivity_onset": sensitivity_onset,
            # "sensitivity_offset": sensitivity_offset,
            # "ppv_onset": ppv_onset,
            # "ppv_offset": ppv_offset,
            # "f1_onset": f1_onset,
            # "f1_offset": f1_offset,
        }

    # Onset and Offset Extraction
    def onset_offset_extraction(self, y_true, y_pred):

        onsets_true = []
        onsets_pred = []
        offsets_true = []
        offsets_pred = []

        # For each category (fiducial), calculate onsets and offsets and save them
        # in a list, with 6 lists inside the list
        for idx, cat in enumerate(range(6)):
            y_t = y_true[cat]
            y_p = y_pred[cat]

            onset_true = (y_t[1:] > y_t[:-1]) & (y_t[1:] == 1)
            onset_pred = (y_p[1:] > y_p[:-1]) & (y_p[1:] == 1)
            onsets_true.append(np.where(onset_true)[0] + 1)     # +1 to align with original indices
            onsets_pred.append(np.where(onset_pred)[0] + 1)

            offset_true = (y_t[1:] < y_t[:-1]) & (y_t[1:] == 0)
            offset_pred = (y_p[1:] < y_p[:-1]) & (y_p[1:] == 0)
            offsets_true.append(np.where(offset_true)[0] + 1)
            offsets_pred.append(np.where(offset_pred)[0] + 1)

        return onsets_true, onsets_pred, offsets_true, offsets_pred

    # Greedy matching algorithm to pair true and predicted onsets/offsets
    def greedyMatchingAlgorithm(self, onsets_true, onsets_pred, offsets_true, offsets_pred):

        matched_onsets = []
        matched_offsets = []
        
        # Match onsets
        for true_onset in onsets_true:
            if len(onsets_pred) == 0:
                break
            closest_pred = min(onsets_pred, key=lambda x: abs(x - true_onset))
            matched_onsets.append((true_onset, closest_pred))
            onsets_pred.remove(closest_pred)
        
        # Match offsets
        for true_offset in offsets_true:
            if len(offsets_pred) == 0:
                break
            closest_pred = min(offsets_pred, key=lambda x: abs(x - true_offset))
            matched_offsets.append((true_offset, closest_pred))
            offsets_pred.remove(closest_pred)
        
        return matched_onsets, matched_offsets 

    # Plot onset and offset value histograms
    def plot_onset_offset_histograms(self, value_range_ms=(-250, 250), fs=512, feature_names=None):

        plt.figure(figsize=(18, 10))
        for category in range(6):

            onset_attr = f"onset_differences_ms_category_{category}"
            offset_attr = f"offset_differences_ms_category_{category}"

            onset_data = getattr(self, onset_attr, [])
            offset_data = getattr(self, offset_attr, [])

            onset_data = np.concatenate(onset_data) if len(onset_data) > 0 else np.array([])
            offset_data = np.concatenate(offset_data) if len(offset_data) > 0 else np.array([])

            onset_time = np.array([v * (1000/fs) for v in onset_data.tolist()])
            offset_time = np.array([v * (1000/fs) for v in offset_data.tolist()])

            # Filter out everything above +/- value_range, because those values are not feasible and probably existant due to mismatched GT/Prediction pairs
            onset_filtered = onset_time[(onset_time >= value_range_ms[0]) & (onset_time <= value_range_ms[1])]
            offset_filtered = offset_time[(offset_time >= value_range_ms[0]) & (offset_time <= value_range_ms[1])]

            # onset_filtered = onset_time
            # offset_filtered = offset_time

            var_onset = np.var(onset_filtered) if len(onset_filtered) > 0 else 0
            std_onset = np.std(onset_filtered) if len(onset_filtered) > 0 else 0

            var_offset = np.var(offset_filtered) if len(offset_filtered) > 0 else 0
            std_offset = np.std(offset_filtered) if len(offset_filtered) > 0 else 0

            onset_bins = int(np.ceil(np.sqrt(len(onset_filtered)))) if len(onset_filtered) > 0 else 1
            offset_bins = int(np.ceil(np.sqrt(len(offset_filtered)))) if len(offset_filtered) > 0 else 1

            # Onset plot
            plt.subplot(2, 6, category + 1)
            plt.hist(onset_filtered, bins=onset_bins, color='skyblue', edgecolor='black', range=value_range_ms)
            title = f'Onset Cat {category}'
            if feature_names:
                title = f'Onset {feature_names[category]}'
            # Plot vertical lines dependent on current tolerance
            plt.axvline(x=self.tol_ms, color='red', linestyle='--', alpha=0.7, label='Tolerance')
            plt.axvline(x=-self.tol_ms, color='red', linestyle='--', alpha=0.7)
            # Shade the area between the tolerance lines
            plt.fill_betweenx(y=[0, plt.ylim()[1]], x1=self.tol_ms, x2=-self.tol_ms, color='red', alpha=0.1)
            # Calculate the percentage of values within tolerances compared to all data
            if len(onset_filtered) > 0:
                within_tolerance = onset_filtered[(onset_filtered <= self.tol_ms) & (onset_filtered >= -self.tol_ms)]
                percentage_within = (len(within_tolerance) / len(onset_filtered)) * 100
            else:
                percentage_within = 0
            # Add information centered above the plot
            plt.text(0, plt.ylim()[1] * 0.99, f'Within Tolerance: {percentage_within:.2f}%', ha='center', va='top', fontsize=10, color='red')
            # Add the title and labels
            plt.title(f'{title}\nVar: {var_onset:.2f} $ms^2$\nStd: {std_onset:.2f} $ms$', fontsize=10)
            plt.xlabel('Diff [ms]')
            plt.ylabel('Count')

            # Offset plot
            plt.subplot(2, 6, category + 7)
            plt.hist(offset_filtered, bins=offset_bins, color='salmon', edgecolor='black', range=value_range_ms)
            title = f'Offset Cat {category}'
            if feature_names:
                title = f'Offset {feature_names[category]}'
            plt.axvline(x=self.tol_ms, color='red', linestyle='--', alpha=0.7, label='Tolerance')
            plt.axvline(x=-self.tol_ms, color='red', linestyle='--', alpha=0.7)
            plt.fill_betweenx(y=[0, plt.ylim()[1]], x1=self.tol_ms, x2=-self.tol_ms, color='red', alpha=0.1)
            if len(offset_filtered) > 0:
                within_tolerance = offset_filtered[(offset_filtered <= self.tol_ms) & (offset_filtered >= -self.tol_ms)]
                percentage_within = (len(within_tolerance) / len(offset_filtered)) * 100
            else:
                percentage_within = 0
            plt.text(0, plt.ylim()[1] * 0.99, f'Within Tolerance: {percentage_within:.2f}%', ha='center', va='top', fontsize=10, color='red')
            plt.title(f'{title}\nVar: {var_offset:.2f} $ms^2$\nStd: {std_offset:.2f} $ms$', fontsize=10)
            plt.xlabel('Diff [ms]')
            plt.ylabel('Count')

        plt.suptitle(f'Onset/Offset Differences - Sampling Rate: {self.sampling_rate} Hz, Tolerance: {self.tol_ms} ms', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Preparation for CDF Plot (not working yet)
        # for category in range(6):

        #     onset_attr = f"onset_differences_ms_category_{category}"
        #     offset_attr = f"offset_differences_ms_category_{category}"

        #     onset_data = getattr(self, onset_attr, [])
        #     offset_data = getattr(self, offset_attr, [])

        #     onset_data = np.concatenate(onset_data) if len(onset_data) > 0 else np.array([])
        #     offset_data = np.concatenate(offset_data) if len(offset_data) > 0 else np.array([])

        #     onset_time = np.array([v * (1000/fs) for v in onset_data.tolist()])
        #     offset_time = np.array([v * (1000/fs) for v in offset_data.tolist()])

        #     # Filter out everything above +/- value_range, because those values are not feasible and probably existant due to mismatched GT/Prediction pairs
        #     onset_filtered = onset_time[(onset_time >= value_range_ms[0]) & (onset_time <= value_range_ms[1])]
        #     offset_filtered = offset_time[(offset_time >= value_range_ms[0]) & (offset_time <= value_range_ms[1])]

        #     # Plot a CDF (cumulative distribution function) plot of the errors using the given tolerance
        #     plt.subplot(2, 2, 1)
        #     errors = np.sort(np.abs(onset_filtered))
        #     cdf = np.arange(1, len(errors) + 1) / len(errors) if len(errors) > 0 else np.array([])
        #     plt.figure(figsize=(16, 10))
        #     plt.plot(errors, cdf, label="CDF of values outside tolerance")
        #     prop = np.mean(errors <= self.tol_ms) * 100 if len(errors) > 0 else 0
        #     plt.axvline(x=self.tol_ms, color='red', linestyle='--', alpha=0.7, label='Tolerance')
        #     plt.text(self.tol_ms, 0.05, f'Tolerance ({self.tol_ms} ms)', rotation=90, color='red', va='center', ha='right')
        #     plt.xlabel('Absolute Error [ms]')
        #     plt.ylabel('Cumulative Probability')
        #     plt.title(f'Cumulative Distribution Function (CDF) of Onset Errors\nPercentage within Tolerance: {prop:.2f}%', fontsize=14)

        #     # now for offset
        #     plt.subplot(2, 2, 2)
        #     errors = np.sort(np.abs(offset_filtered))
        #     cdf = np.arange(1, len(errors) + 1) / len(errors) if len(errors) > 0 else np.array([])
        #     plt.figure(figsize=(16, 10))
        #     plt.plot(errors, cdf, label="CDF of values outside tolerance")
        #     prop = np.mean(errors <= self.tol_ms) * 100 if len(errors) > 0 else 0
        #     plt.axvline(x=self.tol_ms, color='red', linestyle='--', alpha=0.7, label='Tolerance')
        #     plt.text(self.tol_ms, 0.05, f'Tolerance ({self.tol_ms} ms)', rotation=90, color='red', va='center', ha='right')
        #     plt.xlabel('Absolute Error [ms]')
        #     plt.ylabel('Cumulative Probability')
        #     plt.title(f'Cumulative Distribution Function (CDF) of Offset Errors\nPercentage within Tolerance: {prop:.2f}%', fontsize=14)

        #     plt.legend()
        #     plt.grid()
        #     plt.show()

# Convolution + BatchNorm + ReLU Block (conbr)
# The order of Relu and Batchnorm is interchangeable and influences the performance and training speed
class conbr_block(nn.Module):
    """
    A convolutional block consisting of Conv1d, ReLU activation, and BatchNorm1d.
    
    This block applies a 1D convolution followed by ReLU activation and batch normalization,
    which is a common pattern in convolutional neural networks for feature extraction.
    
    Args:
        in_channels (int): Number of input channels for the Conv1d layer.
        out_channels (int): Number of output channels for the Conv1d layer.
        kernel_size (int or tuple): Size of the convolutional kernel.
        stride (int or tuple): Stride of the convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """
        Initializes the conbr_block module.
        Args:
            in_channels (int): Number of input channels for the Conv1d layer.
            out_channels (int): Number of output channels for the Conv1d layer.
            kernel_size (int or tuple): Size of the convolutional kernel.
            stride (int or tuple): Stride of the convolution.
        The block consists of a 1D convolutional layer followed by a ReLU activation and batch normalization.
        """
        super(conbr_block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same', bias=True),
            # nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        return self.net(x)

# U-Net 1D Model
class UNET_1D(pl.LightningModule):

    def __init__(self, in_channels, layer_n, out_channels=1, kernel_size=3):
        super(UNET_1D, self).__init__()

        self.save_hyperparameters()

        # self.loss = nn.BCEWithLogitsLoss() # 20250214_04
        # self.loss = nn.CrossEntropyLoss() # 20250214_03
        self.loss = nn.BCELoss() # 20250214_05 # 20250215_01 # 20250215_02 # 20250221_01

        # Intersection over Union metric
        self.jaccard = BinaryJaccardIndex(threshold=0.5)

        # Custom metrics
        self.multi_custom_metrics = MultiToleranceWrapper(tol_ms=[5, 10, 40, 150], sampling_rate=512)

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

        x_hat = self.forward(x)

        # Logg loss 
        loss = self.loss(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Update Jaccard metric / intersection over Union
        self.log('train_jaccard', self.jaccard(self.postprocess_prediction(x_hat), self.postprocess_ground_truth(y)), prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self.forward(x)

        loss = self.loss(x_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.jaccard.update(self.postprocess_prediction(x_hat), self.postprocess_ground_truth(y))
        self.multi_custom_metrics.update(self.postprocess_prediction(x_hat), self.postprocess_ground_truth(y))

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self.forward(x)

        loss = self.loss(x_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Update Jaccard metric / intersection over Union
        self.log('val_jaccard', self.jaccard(self.postprocess_prediction(x_hat), self.postprocess_ground_truth(y)), prog_bar=True, logger=True)

        return loss

    def on_test_epoch_end(self):

        # Compute intersection over Union
        self.jaccard.compute()
        self.log('test_jaccard', self.jaccard, prog_bar=True, logger=True)
        self.jaccard.reset()

        # Compute custom metrics
        results = self.multi_custom_metrics.compute()
        self.log_dict(results, prog_bar=True, logger=True)
        self.multi_custom_metrics.reset()

        print('Test Epoch End')
        print('-----------------------------------')

    def postprocess_prediction(self, y):
        # Check if y is batched (shape: [batch_size, channels, length])
        if y.dim() == 3:
            # Iterate over batch
            for i in range(y.shape[0]):
                y_ = (y[i] > 0.5).float()
                y[i, 0] = y_[0]
                y[i, 2] = y_[2]
                y[i, 4] = y_[4]
                # For channels 1, 3, 5, set the local maximums with confidence > 0.7 to 1
                for ch_idx in [1, 3, 5]:
                    high_conf_indices = (y[i, ch_idx] > 0.7).nonzero(as_tuple=True)[0]
                    for idx in high_conf_indices:
                        left = y[i, ch_idx][idx - 1] if idx > 0 else float('-inf')
                        right = y[i, ch_idx][idx + 1] if idx < y[i, ch_idx].shape[0] - 1 else float('-inf')
                        if y[i, ch_idx][idx] >= left and y[i, ch_idx][idx] >= right:
                            y[i, ch_idx].zero_()
                            y[i, ch_idx][idx] = 1.0
        else:
            y_ = (y > 0.5).float()
            y[0] = y_[0]
            y[2] = y_[2]
            y[4] = y_[4]
            for ch_idx in [1, 3, 5]:
                high_conf_indices = (y[ch_idx] > 0.5).nonzero(as_tuple=True)[0]
                for idx in high_conf_indices:
                    left = y[ch_idx][idx - 1] if idx > 0 else float('-inf')
                    right = y[ch_idx][idx + 1] if idx < y[ch_idx].shape[0] - 1 else float('-inf')
                    if y[ch_idx][idx] >= left and y[ch_idx][idx] >= right:
                        y[ch_idx].zero_()
                        y[ch_idx][idx] = 1.0
        return y
    
    def postprocess_ground_truth(self, y):
        # Ground truth: Channels 1, 3, 5, set local maxima where value equals 1
        # Check if y is batched (shape: [batch_size, channels, length])
        if y.dim() == 3:
            # Iterate over batch
            for i in range(y.shape[0]):
                for ch_idx in [1, 3, 5]:
                    true_indices = (y[i, ch_idx] == 1).nonzero(as_tuple=True)[0]
                    y[i, ch_idx].zero_()
                    y[i, ch_idx][true_indices] = 1.0
        else:
            # Single sample
            for ch_idx in [1, 3, 5]:
                true_indices = (y[ch_idx] == 1).nonzero(as_tuple=True)[0]
                y[ch_idx].zero_()
                y[ch_idx][true_indices] = 1.0
        return y    

    def compute_jaccard_index(self, y_true, y_pred):
        jaccard = BinaryJaccardIndex()
        # print(jaccard(y_pred, y_true))
        return jaccard(y_pred, y_true)
    
# Wrapper for UNET Models (crucial for trying diferent architectures without a hassle)
class EKG_Segmentation_Module(pl.LightningModule):

    def __init__(self, model, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR'):
        super().__init__()

        # Model
        self.model = model

        # Hyperparameters
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name

        self.criterion = nn.BCELoss()

        # Save hyperparameters!
        self.save_hyperparameters()

    def configure_optimizers(self):
        """
        Configures and returns the optimizer for training the model.

        Returns:
            torch.optim.Optimizer: An optimizer initialized with specified learning rate and weight decay.
        """
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            return [optimizer], [scheduler]
        elif self.scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor after passing through the model.
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        x_hat = self.forward(x)

        # Logg loss 
        loss = self.loss(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Update Jaccard metric / intersection over Union
        self.log('train_jaccard', self.jaccard(self.postprocess_prediction(x_hat), self.postprocess_ground_truth(y)), prog_bar=True, logger=True)

        return loss