### **`data_loader.py`**

import os
import pickle
import numpy as np
import random
from scipy.interpolate import CubicSpline

# --- Configuration ---
# Update this path to point to your dataset directory
base_dir = r".\data"
ir = 3  # Interpolate interval
before = 2  # Minutes before the event
after = 2  # Minutes after the event


def interpolate_numpy_array(arr, desired_length):
    """Interpolates a 1D numpy array to a desired length using Cubic Spline."""
    if len(arr) == 0:
        return np.zeros(desired_length)
    cs = CubicSpline(np.linspace(0, 1, len(arr)), arr)
    x_new = np.linspace(0, 1, desired_length)
    interpolated_arr = cs(x_new)
    return interpolated_arr


def _process_records(o_records, y_records, z_records, groups_records):
    """Helper function to process a set of records (train, val, or test)."""
    x1_data, x2_data, x3_data = [], [], []
    for i in range(len(o_records)):
        min_distance_list, max_distance_list = o_records[i]

        # Curve interpolation and padding based on z_records offset
        if z_records[i] > 0:
            min_inter = interpolate_numpy_array(min_distance_list, 180 * (after + z_records[i]))
            min_pad = np.zeros(180 * (before + 1 + after - (after + z_records[i])))
            min_dist_inter = np.concatenate((min_inter, min_pad))

            max_inter = interpolate_numpy_array(max_distance_list, 180 * (after + z_records[i]))
            max_pad = np.zeros(180 * (before + 1 + after - (after + z_records[i])))
            max_dist_inter = np.concatenate((max_inter, max_pad))
        elif z_records[i] < 0:
            min_inter = interpolate_numpy_array(min_distance_list, 180 * (before + 1 + after + z_records[i]))
            min_pad = np.zeros(180 * (-z_records[i]))
            min_dist_inter = np.concatenate((min_pad, min_inter))

            max_inter = interpolate_numpy_array(max_distance_list, 180 * (before + 1 + after + z_records[i]))
            max_pad = np.zeros(180 * (-z_records[i]))
            max_dist_inter = np.concatenate((max_pad, max_inter))
        else:
            min_dist_inter = interpolate_numpy_array(min_distance_list, 180 * (before + 1 + after))
            max_dist_inter = interpolate_numpy_array(max_distance_list, 180 * (before + 1 + after))

        x1_data.append([min_dist_inter, max_dist_inter])
        # x2 represents a 3-minute window (180*3=540 -> 720-180)
        x2_data.append([min_dist_inter[180:720], max_dist_inter[180:720]])
        # x3 represents the central 1-minute window
        x3_data.append([min_dist_inter[180 * before:180 * -after], max_dist_inter[180 * before:180 * -after]])

    x1_data = np.array(x1_data, dtype="float32").transpose((0, 2, 1))
    x2_data = np.array(x2_data, dtype="float32").transpose((0, 2, 1))
    x3_data = np.array(x3_data, dtype="float32").transpose((0, 2, 1))
    y_data = np.array(y_records, dtype="float32")

    return x1_data, x2_data, x3_data, y_data, groups_records


def load_data():
    """Loads, preprocesses, and splits the apnea-ecg data."""
    # Path to the preprocessed pickle file
    data_path = os.path.join(base_dir,"data.pkl")
    with open(data_path, 'rb') as f:
        apnea_ecg = pickle.load(f)

    # Process training data and split into training and validation sets
    x_train1_full, x_train2_full, x_train3_full, y_train_full, groups_train_full = _process_records(
        apnea_ecg["o_train"], apnea_ecg["y_train"], apnea_ecg["z_train"], apnea_ecg["groups_train"]
    )

    # Create indices and shuffle for splitting
    all_indices = list(range(len(y_train_full)))
    random.seed(43)
    random.shuffle(all_indices)

    train_split = int(len(all_indices) * 0.7)
    train_indices = all_indices[:train_split]
    val_indices = all_indices[train_split:]

    # Create training set
    x_training1, x_training2, x_training3 = x_train1_full[train_indices], x_train2_full[train_indices], x_train3_full[
        train_indices]
    y_training, groups_training = y_train_full[train_indices], [groups_train_full[i] for i in train_indices]

    # Create validation set
    x_val1, x_val2, x_val3 = x_train1_full[val_indices], x_train2_full[val_indices], x_train3_full[val_indices]
    y_val, groups_val = y_train_full[val_indices], [groups_train_full[i] for i in val_indices]

    # Process test data
    x_test1, x_test2, x_test3, y_test, groups_test = _process_records(
        apnea_ecg["o_test"], apnea_ecg["y_test"], apnea_ecg["z_test"], apnea_ecg["groups_test"]
    )

    return (x_training1, x_training2, x_training3, y_training, groups_training,
            x_val1, x_val2, x_val3, y_val, groups_val,
            x_test1, x_test2, x_test3, y_test, groups_test)