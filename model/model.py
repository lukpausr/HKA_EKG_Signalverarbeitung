# model.py

# required imports
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Metric
from torchmetrics.classification import BinaryJaccardIndex
from matplotlib import pyplot as plt

import os
import numpy as np
import pandas as pd
import math

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

        self.path_to_history_data = None

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
        if self.path_to_history_data is not None:
            os.makedirs(self.path_to_history_data, exist_ok=True)

        results = {}
        for name, metric in self.metrics.items():
            computed_metric = metric.compute(data_path = self.path_to_history_data)
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

    def compute(self, data_path = None):

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

        # Save the computed metrics as csv
        if data_path is not None:
            pd.DataFrame({
                "tp_onset_global": [tp_onset_global],
                "fp_onset_global": [fp_onset_global],
                "fn_onset_global": [fn_onset_global],
                "tp_offset_global": [tp_offset_global],
                "fp_offset_global": [fp_offset_global],
                "fn_offset_global": [fn_offset_global],
                "sensitivity_onset_global": [sensitivity_onset_global],
                "sensitivity_offset_global": [sensitivity_offset_global],
                "ppv_onset_global": [ppv_onset_global],
                "ppv_offset_global": [ppv_offset_global],
                "f1_onset_global": [f1_onset_global],
                "f1_offset_global": [f1_offset_global],
            }).to_csv(os.path.join(data_path, f"computed_metrics_tol_{self.tol_ms}ms.csv"), index=False)

        self.plot_onset_offset_histograms(data_path=data_path)

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
    def plot_onset_offset_histograms(self, value_range_ms=(-250, 250), fs=512, feature_names=None, data_path=None):

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

            if(category == 1):
                print(onset_filtered, offset_filtered)

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

            # Save raw difference data and metrics data as separate files
            if data_path is not None:

                # Save onset differences
                onset_data = pd.DataFrame({
                    "onset_differences_ms": onset_filtered
                })
                onset_data.to_csv(os.path.join(data_path, f"onset_differences_cat_{category}_tol_{self.tol_ms}ms.csv"), index=False)
                
                # Save offset differences
                offset_data = pd.DataFrame({
                    "offset_differences_ms": offset_filtered
                })
                offset_data.to_csv(os.path.join(data_path, f"offset_differences_cat_{category}_tol_{self.tol_ms}ms.csv"), index=False)
                
                # Save computed metrics data
                metrics_data = pd.DataFrame({
                    "category": [category],
                    "var_onset": [var_onset],
                    "std_onset": [std_onset],
                    "var_offset": [var_offset],
                    "std_offset": [std_offset],
                    "tolerance_ms": [self.tol_ms],
                    "n_onset_samples": [len(onset_filtered)],
                    "n_offset_samples": [len(offset_filtered)]
                })
                metrics_data.to_csv(os.path.join(data_path, f"metrics_cat_{category}_tol_{self.tol_ms}ms.csv"), index=False)

        plt.suptitle(f'Onset/Offset Differences - Sampling Rate: {self.sampling_rate} Hz, Tolerance: {self.tol_ms} ms', fontsize=14)
        plt.tight_layout()

        # Save the plot at the given path
        if data_path is not None:
            plt.savefig(os.path.join(data_path, f"computed_metrics_histogram_tol_{self.tol_ms}ms.png"))
        
        plt.show()
        plt.close()

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

# Wrapper for UNET Models (crucial for trying diferent architectures without a hassle)
class EKG_Segmentation_Module(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, optimizer_name='Adam', weight_decay=0.0, scheduler_name='StepLR', model_name=None):
        super().__init__()

        # Meta information
        self.model_name = "parent_class" # model_name if model_name is not None else self.__class__.__name__
        self.config = None

        # Hyperparameters
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name

        self.criterion = nn.BCEWithLogitsLoss()

        # Save hyperparameters!
        self.save_hyperparameters()

        # Metrics
        self.train_jaccard = BinaryJaccardIndex()
        self.val_jaccard = BinaryJaccardIndex()
        self.test_jaccard = BinaryJaccardIndex()

        self.multi_tolerance_metrics = MultiToleranceWrapper()

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
        return NotImplementedError("Forward method not implemented. Please implement the forward method in the subclass.")
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self.forward(x)

        # Log loss (BCEWithLogits integrates BCE and sigmoid to calculate loss)
        loss = self.criterion(x_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Apply sigmoid activation
        x_hat = nn.functional.sigmoid(x_hat)

        # Plot some example predictions
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            x_np = x[0].detach().cpu()
            y_np = y[0].detach().cpu()
            x_hat_np = x_hat[0].detach().cpu()
            self.generatePlot(x_np, y_np, x_np, x_hat_np, config=self.config, path=None)
            x_hat_np = self.postprocess_prediction(x_hat_np)
            y_np = self.postprocess_ground_truth(y_np)
            self.generatePlot(x_np, y_np, x_np, x_hat_np, config=self.config, path=None)

        # Log Jaccard metric / intersection over Union
        # self.train_jaccard.update(self.postprocess_prediction(x_hat), self.postprocess_ground_truth(y))
        # self.log('train_jaccard', self.train_jaccard, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self.forward(x)

        # Log loss (BCEWithLogits integrates BCE and sigmoid to calculate loss)
        loss = self.criterion(x_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Apply sigmoid activation
        x_hat = nn.functional.sigmoid(x_hat)

        # Log Jaccard metric / intersection over Union
        self.val_jaccard.update(self.postprocess_prediction(x_hat), self.postprocess_ground_truth(y))
        self.log('val_jaccard', self.val_jaccard, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self.forward(x)

        # Log loss (BCEWithLogits integrates BCE and sigmoid to calculate loss)
        loss = self.criterion(x_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Apply sigmoid activation
        x_hat = nn.functional.sigmoid(x_hat)

        # Perform postprocessing
        x_hat_processed = self.postprocess_prediction(x_hat)
        y_processed = self.postprocess_ground_truth(y)

        # Log Jaccard metric / intersection over Union
        self.test_jaccard.update(x_hat_processed, y_processed)
        self.log('test_jaccard', self.test_jaccard, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log custom metrics for different tolerance levels
        self.multi_tolerance_metrics.update(x_hat_processed, y_processed)

        # Print Graphics
        if True:
            x_np = x[0].detach().cpu()
            y_np = y[0].detach().cpu()
            x_hat_np = x_hat[0].detach().cpu()
            self.generatePlot(x_np, y_np, x_np, x_hat_np, config=self.config, path=os.path.join(self.multi_tolerance_metrics.path_to_history_data, f"raw_{batch_idx}.png") if self.multi_tolerance_metrics.path_to_history_data else None)
            x_hat_np = self.postprocess_prediction(x_hat_np)
            y_np = self.postprocess_ground_truth(y_np)
            self.generatePlot(x_np, y_np, x_np, x_hat_np, config=self.config, path=os.path.join(self.multi_tolerance_metrics.path_to_history_data, f"postprocessed_{batch_idx}.png") if self.multi_tolerance_metrics.path_to_history_data else None)

        return loss
    
    def on_train_epoch_end(self):
        # self.train_jaccard.reset()
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        
        jaccard_index = self.val_jaccard.compute()
        self.log('val_jaccard', jaccard_index, prog_bar=True, logger=True)

        self.val_jaccard.reset()

        return super().on_validation_epoch_end()

    def on_test_epoch_end(self):

        jaccard_index = self.test_jaccard.compute()
        self.log('test_jaccard', jaccard_index, prog_bar=True, logger=True)

        results = self.multi_tolerance_metrics.compute()
        self.log_dict(results, prog_bar=True, logger=True)

        self.test_jaccard.reset()
        self.multi_tolerance_metrics.reset()

        print('Test Epoch End')
        print('-----------------------------------')

        return super().on_test_epoch_end()

    # This method was generated/optimized using Claude Sonnet 4
    def postprocess_prediction(self, y):
        """
        Fully GPU-optimized postprocessing for predictions.
        """
        y_processed = y.clone()
        
        if y.dim() == 3:  # Batched
            # Threshold channels 0, 2, 4 (binary classification) - VECTORIZED
            y_processed[:, [0, 2, 4]] = (y[:, [0, 2, 4]] > 0.5).float()
            
            # Process channels 1, 3, 5 (peak detection)
            for ch_idx in [1, 3, 5]:
                if self.config['use_nms']:
                    # GPU-OPTIMIZED: Vectorized NMS across ALL batches
                    window_size = self.config['nms_window_size']
                    peaks_batch = self.batch_non_maximum_suppression_1d(
                        y[:, ch_idx], window_size
                    )
                    y_processed[:, ch_idx] = peaks_batch.float()
                else:
                    # Original vectorized local maxima detection
                    peak_channel = y[:, ch_idx]  # Shape: [batch, length]
                    
                    # High confidence mask
                    high_conf = peak_channel > 0.5
                    
                    # Compute local maxima (vectorized across all batches)
                    padded = torch.nn.functional.pad(peak_channel, (1, 1), mode='constant', value=float('-inf'))
                    left = padded[:, :-2]
                    center = padded[:, 1:-1]
                    right = padded[:, 2:]
                    
                    is_local_max = (center >= left) & (center >= right)
                    final_mask = high_conf & is_local_max
                    
                    # Zero out and set peaks
                    y_processed[:, ch_idx] = 0.0
                    y_processed[:, ch_idx][final_mask] = 1.0
                        
        else:  # Single sample
            # Threshold channels 0, 2, 4
            y_processed[[0, 2, 4]] = (y[[0, 2, 4]] > 0.5).float()
            
            # Process channels 1, 3, 5
            for ch_idx in [1, 3, 5]:
                if self.config['use_nms']:
                    window_size = self.config['nms_window_size']

                    # Convert single sample to batch format
                    signal_batch = y[ch_idx].unsqueeze(0)  # Add batch dimension
                    peaks_batch = self.batch_non_maximum_suppression_1d(signal_batch, window_size)
                    y_processed[ch_idx] = peaks_batch[0].float()  # Remove batch dimension
                else:
                    # Original method
                    peak_channel = y[ch_idx]
                    high_conf = peak_channel > 0.5
                    
                    padded = torch.nn.functional.pad(peak_channel.unsqueeze(0), (1, 1), mode='constant', value=float('-inf'))
                    left = padded[:, :-2].squeeze(0)
                    center = padded[:, 1:-1].squeeze(0)
                    right = padded[:, 2:].squeeze(0)
                    
                    is_local_max = (center >= left) & (center >= right)
                    final_mask = high_conf & is_local_max
                    
                    y_processed[ch_idx] = 0.0
                    y_processed[ch_idx][final_mask] = 1.0
            
        return y_processed

    # This method was generated/optimized using Claude Sonnet 4
    def batch_non_maximum_suppression_1d(self, signals, window_size=20):
        """
        Ultra-optimized vectorized NMS for entire batch without any loops.
        """
        batch_size, signal_length = signals.shape
        device = signals.device
        
        # Find all potential peaks (vectorized across batch)
        padded = torch.nn.functional.pad(signals, (1, 1), mode='constant', value=float('-inf'))
        left = padded[:, :-2]
        center = padded[:, 1:-1] 
        right = padded[:, 2:]
        
        is_local_max = (center >= left) & (center >= right) & (center > 0.5)
        
        # Create batch result tensor
        batch_peaks = torch.zeros_like(signals, dtype=torch.bool, device=device)
        
        # For each batch, apply vectorized suppression
        for batch_idx in range(batch_size):
            candidates = is_local_max[batch_idx].nonzero(as_tuple=True)[0]
            
            if len(candidates) == 0:
                continue
            
            # Sort by confidence
            confidences = signals[batch_idx, candidates]
            sorted_indices = torch.argsort(confidences, descending=True)
            sorted_candidates = candidates[sorted_indices]
            
            # Vectorized suppression matrix
            if len(sorted_candidates) > 0:
                pos_diff = sorted_candidates.unsqueeze(1) - sorted_candidates.unsqueeze(0)
                distances = torch.abs(pos_diff)
                suppresses = (distances <= window_size // 2) & (distances > 0)
                mask = torch.triu(torch.ones_like(suppresses), diagonal=1)
                suppresses = suppresses & mask
                is_suppressed = torch.any(suppresses, dim=0)
                final_candidates = sorted_candidates[~is_suppressed]
                
                if len(final_candidates) > 0:
                    batch_peaks[batch_idx, final_candidates] = True
        
        return batch_peaks

    def postprocess_ground_truth(self, y):
        """
        Corrected GPU-vectorized postprocessing for ground truth.
        """
        # Clone to avoid modifying input tensor
        y_processed = y.clone()
        
        if y.dim() == 3:  # Batched tensor [batch_size, channels, length]
            # Process each channel individually to avoid indexing issues
            for ch_idx in [1, 3, 5]:
                # Find where values equal 1 for this specific channel
                true_mask = (y[:, ch_idx] == 1)  # Shape: [batch_size, length]
                
                # Zero out the channel
                y_processed[:, ch_idx] = 0.0
                
                # Set back to 1 where original was 1
                y_processed[:, ch_idx][true_mask] = 1.0
                
        else:  # Single sample tensor [channels, length]
            # Process each channel individually
            for ch_idx in [1, 3, 5]:
                # Find where values equal 1 for this specific channel
                true_mask = (y[ch_idx] == 1)  # Shape: [length]
                
                # Zero out the channel
                y_processed[ch_idx] = 0.0
                
                # Set back to 1 where original was 1
                y_processed[ch_idx][true_mask] = 1.0
        
        return y_processed

    # Function to generate a plot of the ECG data and the labels
    def generatePlot(self, x, y, x_hat, y_hat, feature_names=['P wave', 'P peak', 'QRS complex', 'R peak', 'T wave', 'T peak'], lead_names=None, config=None, path=None):
        
        if config is not None:
            feature_names=config['feature_names'] if 'feature_names' in config else feature_names
            lead_names=config['data_cols'] if 'data_cols' in config else lead_names
        
        fig, axs = plt.subplots(7, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [5, 1, 1, 1, 1, 1, 1]})
        # Plot ECG data
        # Plot all ECG leads if multiple leads exist
        if x.shape[0] > 1:
            for lead_idx in range(x.shape[0]):
                axs[0].plot(x[lead_idx].numpy(), label=f'Lead {lead_idx + 1}' if lead_names is None else lead_names[lead_idx])
                axs[0].legend()
        else:
            axs[0].plot(x[0].numpy(), color='blue')

        axs[0].set_title('ECG Data')
        # Range between min and max value of ECG data
        # Get min and max values across all leads
        all_leads_min = min([lead.numpy().min() for lead in x])
        all_leads_max = max([lead.numpy().max() for lead in x])
        axs[0].set_ylim([all_leads_min*1.1, all_leads_max*1.1])
        axs[0].set_ylabel('Amplitude [mV]')

        # Calculate appropriate tick ranges based on data
        y_min = all_leads_min * 1.1
        y_max = all_leads_max * 1.1

        # Major ticks every 0.5mV
        major_ticks = np.arange(
            math.floor(y_min / 0.5) * 0.5,  # Round down to nearest 0.5mV
            math.ceil(y_max / 0.5) * 0.5 + 0.5,  # Round up to nearest 0.5mV
            0.5  # Step size: 0.5mV
        )
        
        # Minor ticks every 0.1mV
        minor_ticks = np.arange(
            math.floor(y_min / 0.1) * 0.1,  # Round down to nearest 0.1mV
            math.ceil(y_max / 0.1) * 0.1 + 0.1,  # Round up to nearest 0.1mV
            0.1  # Step size: 0.1mV
        )

        # Set the major and minor ticks
        axs[0].set_yticks(major_ticks)  # Major ticks at 0.5mV intervals
        axs[0].set_yticks(minor_ticks, minor=True)  # Minor ticks at 0.1mV intervals
        
        axs[0].set_xticks(range(0, 551, 100)) # Major ticks every 100 samples
        axs[0].set_xticks(range(0, 551, 20), minor=True) # Minor ticks every 20 samples

        # Medical ECG paper-style grid
        axs[0].grid(which='major', color='lightgray', linestyle='-', linewidth=0.8)  # 0.5mV lines
        axs[0].grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # 0.1mV lines
        axs[0].minorticks_on()
        axs[0].set_axisbelow(True)

        # Optional: Add grid labels for better readability
        axs[0].tick_params(axis='y', which='major', labelsize=10, color='gray')
        axs[0].tick_params(axis='y', which='minor', labelsize=8, color='lightgray')

        # Plot labels
        for i in range(6):
            axs[i + 1].plot(y[i].numpy(), color='green')
            axs[i + 1].plot(y_hat[i].numpy(), color='blue')
            # axs[i + 1].plot(abs(y[i].numpy() - y_hat[i].numpy()), color='red', linestyle='--')
            axs[i + 1].set_ylim([-0.1, 1.1])
            axs[i + 1].set_ylabel(feature_names[i])
            axs[i + 1].set_yticks([0, 1])
            axs[i + 1].set_yticklabels(['0', '1'])
            axs[i + 1].set_xticks([])
            axs[i + 1].legend(['Ground truth', 'Prediction'], fontsize='x-small', fancybox=False, loc='upper right')

        axs[-1].set_xlabel('Sample [n]')
        axs[-1].set_xticks(range(0, 551, 50))

        plt.tight_layout()

        if path is not None and os.path.exists(os.path.dirname(path)):
            plt.savefig(path)
        
        #save_img_path = r"\\nas-k2\homes\Lukas Pelz\10_Arbeit_und_Bildung\20_Masterstudium\01_Semester\90_Projekt\10_DEV\HKA_EKG_Signalverarbeitung\images"
        #os.listdir(save_img_path)
        #number_of_images = len(os.listdir(save_img_path))
        #plt.savefig(save_img_path + r"\plot_" + str(number_of_images) + ".png")
        
        # plt.show()
        plt.close()

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
class UNET_1D(EKG_Segmentation_Module):

    def __init__(
            self,
            in_channels,
            layer_n,
            out_channels=1,
            kernel_size=3,
            learning_rate=1e-3,
            optimizer_name='Adam',
            weight_decay=0.0,
            scheduler_name='StepLR',
            model_name=None
            ):

        # Call parent constructor with optimizer parameters
        super().__init__(learning_rate, optimizer_name, weight_decay, scheduler_name, model_name)

        self.model_name = model_name if model_name is not None else self.__class__.__name__

        # Save hyperparameters
        self.save_hyperparameters()

        # self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.BCELoss()

        # Example input for model
        self.example_input_array = torch.rand(1, in_channels, layer_n)

        # Model parameters
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
        if(self.in_channels != 1):
            self.layer1 = nn.Sequential(
                conbr_block(self.in_channels, 64 * factor, self.kernel_size, stride=1),
                conbr_block(64 * factor, 64 * factor, self.kernel_size, stride=1),
            )
        else:
            self.layer1 = nn.Sequential(
                conbr_block(self.in_channels, self.in_channels * 64 * factor, self.kernel_size, stride=1),
                conbr_block(self.in_channels * 64 * factor, self.in_channels * 64 * factor, self.kernel_size, stride=1),
        )

        self.in_channels = 1  # Reset in_channels to 1 after first layer if it was different

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

        # Sigmoid activation must be applied individually afterwards
        x_hat = in_0

        return x_hat
    

        