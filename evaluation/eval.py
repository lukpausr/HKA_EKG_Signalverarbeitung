import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Overlap quality: Intersection over Union (done, see model.py)

# Boundary accuracy: OnSet/OffSet absolute error (in progress)
# Boundary accuracy: Statistics over dataset (in progress)

# Primary metrics (using given tolerance window)
# Sensitivity (Recall)
# Positive Predictive Value (Precision)
# F1 Score

# Localization accuracy:
# Mittelwert
# Median
# StdDev
# 95-perzentil
# Fehlerverteilung

# Latency: Average inference time per sample

# Reporting includes:
# Dataset, Leads, Reference, Tolerance
# Sensitivity, Precision, F1 Score
# Localization accuracy metrics
# Latency
# Plots

def greedyMatchingAlgorithm(onsets_true, onsets_pred, offsets_true, offsets_pred):
    # Greedy matching algorithm to pair true and predicted onsets/offsets
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

def deviationPerCategory(y_true, y_pred, print_debug=False):
    # Calculate the distance in time between onset true and predictede values
    # Find the OnSets (positive Flanke) of category [0,2,4] and 
    onsets_true = []
    onsets_pred = []
    offsets_true = []
    offsets_pred = []
    for idx, cat in enumerate(range(6)):
        y_t = y_true[cat].numpy()
        y_p = y_pred[cat].numpy()

        onset_true = (y_t[1:] > y_t[:-1]) & (y_t[1:] == 1)
        onset_pred = (y_p[1:] > y_p[:-1]) & (y_p[1:] == 1)
        onsets_true.append(np.where(onset_true)[0] + 1)  # +1 to align with original indices
        onsets_pred.append(np.where(onset_pred)[0] + 1)

        offset_true = (y_t[1:] < y_t[:-1]) & (y_t[1:] == 0)
        offset_pred = (y_p[1:] < y_p[:-1]) & (y_p[1:] == 0)
        offsets_true.append(np.where(offset_true)[0] + 1)
        offsets_pred.append(np.where(offset_pred)[0] + 1)

        if print_debug:
            print(f"Category {cat} - True Onsets: {onsets_true[idx]}")
            print(f"Category {cat} - Predicted Onsets: {onsets_pred[idx]}")
            print(f"Category {cat} - True Offsets: {offsets_true[idx]}")
            print(f"Category {cat} - Predicted Offsets: {offsets_pred[idx]}")

    # Calculate onset and offset differences for each category
    onset_diffs_all = []
    offset_diffs_all = []
    filtered_onset_diffs_all = []
    filtered_offset_diffs_all = []

    tolerance_ms = 20   # Tolerance window in milliseconds
    fs = 512            # Sampling frequency in Hz

    for cat in range(6):
        true_onsets = onsets_true[cat].tolist()
        pred_onsets = onsets_pred[cat].tolist()
        true_offsets = offsets_true[cat].tolist()
        pred_offsets = offsets_pred[cat].tolist()

        # Use greedy matching algorithm for onsets and offsets
        matched_onsets, matched_offsets = greedyMatchingAlgorithm(true_onsets.copy(), pred_onsets.copy(), true_offsets.copy(), pred_offsets.copy())

        onset_differences = np.array([pred - true for true, pred in matched_onsets]) if matched_onsets else np.array([]).tolist()
        offset_differences = np.array([pred - true for true, pred in matched_offsets]) if matched_offsets else np.array([]).tolist()

        onset_differences_ms = np.array([diff * (1000 / fs) for diff in onset_differences])
        offset_differences_ms = np.array([diff * (1000 / fs) for diff in offset_differences])

        # Calculate metrics TP, FP, FN and filter using given tolerance
        tp_onset = np.sum((onset_differences_ms >= -tolerance_ms) & (onset_differences_ms <= tolerance_ms))
        fp_onset = np.sum(onset_differences_ms > tolerance_ms)
        fn_onset = np.sum(onset_differences_ms < -tolerance_ms)
        tp_offset = np.sum((offset_differences_ms >= -tolerance_ms) & (offset_differences_ms <= tolerance_ms))
        fp_offset = np.sum(offset_differences_ms > tolerance_ms)
        fn_offset = np.sum(offset_differences_ms < -tolerance_ms)
        sensitivity_onset = tp_onset / (tp_onset + fn_onset) if (tp_onset + fn_onset) > 0 else 0
        sensitivity_offset = tp_offset / (tp_offset + fn_offset) if (tp_offset + fn_offset) > 0 else 0
        ppv_onset = tp_onset / (tp_onset + fp_onset) if (tp_onset + fp_onset) > 0 else 0
        ppv_offset = tp_offset / (tp_offset + fp_offset) if (tp_offset + fp_offset) > 0 else 0
        f1_onset = 2 * (ppv_onset * sensitivity_onset) / (ppv_onset + sensitivity_onset) if (ppv_onset + sensitivity_onset) > 0 else 0
        f1_offset = 2 * (ppv_offset * sensitivity_offset) / (ppv_offset + sensitivity_offset) if (ppv_offset + sensitivity_offset) > 0 else 0

        if print_debug:
            print(f"Category {cat} - Onset Differences: {onset_differences}")
            print(f"Category {cat} - Offset Differences: {offset_differences}")
            print(f"Category {cat} - Onset TP: {tp_onset}, FP: {fp_onset}, FN: {fn_onset}")
            print(f"Category {cat} - Offset TP: {tp_offset}, FP: {fp_offset}, FN: {fn_offset}")

        onset_diffs_all.append(onset_differences)
        offset_diffs_all.append(offset_differences)

        # Filter out invalid matches
        filtered_onset_diffs_all.append(onset_differences[onset_differences_ms <= tolerance_ms] if len(onset_differences) > 0 else np.array([]))
        filtered_offset_diffs_all.append(offset_differences[offset_differences_ms <= tolerance_ms] if len(offset_differences) > 0 else np.array([]))

    # Returns an array for each element, containing the values for category 1 at indice 0 up to category 6 at indice 5
    return onsets_true, onsets_pred, offsets_true, offsets_pred, onset_diffs_all, offset_diffs_all, filtered_onset_diffs_all, filtered_offset_diffs_all

def plot_onset_offset_histograms(onset, offset, value_range=(-0.1, 0.1), fs=512, feature_names=None):

    onset_time = [np.array(v) * (1/fs) for v in onset]
    offset_time = [np.array(v) * (1/fs) for v in offset]

    plt.figure(figsize=(18, 10))
    for cat in range(len(onset)):
        onset_filtered = onset_time[cat][(onset_time[cat] >= value_range[0]) & (onset_time[cat] <= value_range[1])]
        offset_filtered = offset_time[cat][(offset_time[cat] >= value_range[0]) & (offset_time[cat] <= value_range[1])]

        var_onset = np.var(onset_filtered) if len(onset_filtered) > 0 else 0
        std_onset = np.std(onset_filtered) if len(onset_filtered) > 0 else 0

        var_offset = np.var(offset_filtered) if len(offset_filtered) > 0 else 0
        std_offset = np.std(offset_filtered) if len(offset_filtered) > 0 else 0

        onset_bins = int(np.ceil(np.sqrt(len(onset_filtered)))) if len(onset_filtered) > 0 else 1
        offset_bins = int(np.ceil(np.sqrt(len(offset_filtered)))) if len(offset_filtered) > 0 else 1

        # Onset plot
        plt.subplot(2, len(onset), cat + 1)
        plt.hist(onset_filtered, bins=onset_bins, color='skyblue', edgecolor='black', range=value_range)
        title = f'Onset Cat {cat}'
        if feature_names:
            title = f'Onset {feature_names[cat]}'
        plt.title(f'{title}\nVar: {var_onset:.5f} $s^2$\nStd: {std_onset:.5f} $s$', fontsize=10)
        plt.xlabel('Diff [s]')
        plt.ylabel('Count')

        # Offset plot
        plt.subplot(2, len(onset), cat + 1 + len(onset))
        plt.hist(offset_filtered, bins=offset_bins, color='salmon', edgecolor='black', range=value_range)
        title = f'Offset Cat {cat}'
        if feature_names:
            title = f'Offset {feature_names[cat]}'
        plt.title(f'{title}\nVar: {var_offset:.5f} $s^2$\nStd: {std_offset:.5f} $s$', fontsize=10)
        plt.xlabel('Diff [s]')
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()