##################################################################
# tinyHLS Copyright (C) 2024 FRAUNHOFER INSTITUTE OF MICROELECTRONIC CIRCUITS AND SYSTEMS (IMS), DUISBURG, GERMANY. 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# As a special exception, you may create a larger work that contains
# part or all of the tinyHLS hardware compiler and distribute that 
# work under the terms of your choice, so long as that work is not 
# itself a hardware compiler or template-based code generator or a 
# modified version thereof. Alternatively, if you modify or re-
# distribute the hardware compiler itself, you may (at your option) 
# remove this special exception, which will cause the hardware compi-
# ler and the resulting output files to be licensed under the GNU 
# General Public License without this special exception.
#
#   $$\     $$\                     $$\   $$\ $$\       $$$$$$\
#   $$ |    \__|                    $$ |  $$ |$$ |     $$  __$$\
# $$$$$$\   $$\ $$$$$$$\  $$\   $$\ $$ |  $$ |$$ |     $$ /  \__|
# \_$$  _|  $$ |$$  __$$\ $$ |  $$ |$$$$$$$$ |$$ |     \$$$$$$\
#   $$ |    $$ |$$ |  $$ |$$ |  $$ |$$  __$$ |$$ |      \____$$\
#   $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |     $$\   $$ |
#   \$$$$  |$$ |$$ |  $$ |\$$$$$$$ |$$ |  $$ |$$$$$$$$\\$$$$$$  |
#    \____/ \__|\__|  \__| \____$$ |\__|  \__|\________|\______/
#                         $$\   $$ |
#                         \$$$$$$  |
#                          \______/
###################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.quantization
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import tf2onnx
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

#import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import json
import re
import tinyhls
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, confusion_matrix

import random
import logging
from datetime import datetime

#Pruning
#from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
#from tensorflow_model_optimization.sparsity.keras import strip_pruning
import tempfile

# Quantization 
from fxpmath import Fxp

# Provided from Felix Wichum 
import wfdb
from pathlib import Path
from typing import Tuple
import glob
from scipy import signal

import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, filtfilt

import tensorflow as tf


# Set the seed for reproducibility
seed_value = 31  # You can choose any integer

# Set the seed for Python's random module
random.seed(seed_value)

# Set the seed for NumPy
np.random.seed(seed_value)

# Set the seed for PyTorch
torch.manual_seed(seed_value)  # For CPU
torch.cuda.manual_seed_all(seed_value)  # For GPU

# Optionally, you can set deterministic behavior in PyTorch (may slow down training)
torch.backends.cudnn.deterministic = True


######################## Helper Functions ##############################
# Define quantization configuration
def prepare_model_for_qat(model):
    model.train()  # Set model to training mode
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')  # Use the FBGEMM backend for QAT
    torch.quantization.prepare_qat(model, inplace=True)  # Prepare the model for QAT
    return model

def extract_weights_torch(model, destination_path):
    ctr = 0
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.numpy().flatten()
            # If the layer has bias, handle it separately
            if 'bias' in name:
                bias = param.data.numpy().flatten()
                b_str = "\n".join(map(str, bias))
                with open(os.path.join(destination_path, f'b{ctr}.txt'), 'w') as f:
                    f.write(b_str)
            else:
                w_str = "\n".join(map(str, weights))
                with open(os.path.join(destination_path, f'w{ctr}.txt'), 'w') as f:
                    f.write(w_str)
                ctr += 1
                
def convert_model_to_json(model, input_shape):
    """
    Export the PyTorch model architecture to a JSON string in the specified format.

    Args:
    - model: The PyTorch model to export.
    - input_shape: Tuple representing the expected input shape (e.g., (None, 512, 2)).

    Returns:
    - json_string: A JSON string representing the model architecture.
    """
    model_json = {
        "class_name": "Sequential",
        "config": {
            "name": "sequential_1",
            "layers": []
        },
        "keras_version": "2.12.0",
        "backend": "tensorflow"
    }

    # Add Input Layer
    # rearanging since TF and Torch do not match. 
    shape = [None, input_shape[2], input_shape[1]]
    input_layer = {
        "class_name": "InputLayer",
        "config": {
            "batch_input_shape": shape, #[None, 75, 1]
            "dtype": "float32",
            "sparse": False,
            "ragged": False,
            "name": "input_layer"
        }
    }
    model_json["config"]["layers"].append(input_layer)

    # Iterate through the model's layers and capture their details
    for layer in model.children():
        layer_info = {
            "class_name": str(type(layer).__name__).replace('Conv1d', 'Conv1D'),  # Change to Conv1D
            "config": {}
        }
        # Capture layer-specific parameters
        if isinstance(layer, nn.Conv1d):
            layer_info["config"].update({
                "name": str(layer),
                "trainable": True,
                "dtype": "float32",
                "filters": layer.out_channels,
                "kernel_size": [layer.kernel_size[0]],  # Convert to list
                "strides": [1],
                "padding": "valid",
                "data_format": "channels_last",
                "dilation_rate": [1],  # Added
                "groups": 1,           # Added
                "activation": "relu",  # Assuming ReLU activation
                "use_bias": True,
                "kernel_initializer": {
                    "class_name": "HeUniform",
                    "config": {"seed": None}
                },
                "bias_initializer": {
                    "class_name": "Zeros",
                    "config": {}
                },
                "kernel_regularizer": None,  # Added
                "bias_regularizer": None,    # Added
                "activity_regularizer": None, # Added
                "kernel_constraint": None,    # Added
                "bias_constraint": None        # Added
            })
        elif isinstance(layer, nn.Linear):
            layer_info["class_name"] = "Dense"  # Change Linear to Dense
            layer_info["config"].update({
                "name": str(layer),
                "trainable": True,
                "dtype": "float32",
                "units": layer.out_features,
                "activation": "relu",  # Assuming ReLU activation
                "use_bias": True,
                "kernel_initializer": {
                    "class_name": "HeUniform",
                    "config": {"seed": None}
                },
                "bias_initializer": {
                    "class_name": "Zeros",
                    "config": {}
                },
                "kernel_regularizer": None,  # Added
                "bias_regularizer": None,    # Added
                "activity_regularizer": None, # Added
                "kernel_constraint": None,    # Added
                "bias_constraint": None        # Added
            })
        elif isinstance(layer, nn.AdaptiveAvgPool1d):
            # Change AdaptiveAvgPool1d to GlobalAveragePooling1D
            layer_info["class_name"] = "GlobalAveragePooling1D"
            layer_info["config"].update({
                "name": "global_average_pooling1d",
                "trainable": True,
                "dtype": "float32",
                "data_format": "channels_last",
                "keepdims": False
            })

        # Add layer to model_json
        model_json["config"]["layers"].append(layer_info)

    # Convert the model architecture to a JSON string
    json_string = json.dumps(model_json, indent=None)
    return json_string
# To verify that your path to the data actually contains data 
def list_files_recursively(path):
    for file in Path(path).rglob('*'):
        print(file)

# Define a low-pass filter function
def low_pass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Normalize and clip function
def normalize_and_clip(data: np.ndarray, clip_min: float = -125, clip_max: float = 125) -> np.ndarray:
    # Clip the data to the specified range
    data = np.clip(data, clip_min, clip_max)
    # Normalize to the range [0, 1]
    return (data - clip_min) / (clip_max - clip_min)

# Extracts the PPG signals from the CSV files 
def extract_signal_from_csv(file_path: str) -> np.ndarray:
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Extract the PPG signal
    ppg_signal = df[' PLETH'].values
    
    # Apply low-pass filter
    ppg_signal = low_pass_filter(ppg_signal, cutoff=5.0, fs=125)  # Adjust cutoff and fs as necessary
    
    # Normalize and clip the PPG signal
    ppg_signal = normalize_and_clip(ppg_signal)
    
    return ppg_signal

# Extracts the pulse values from the CSV files 
def extract_pulse_from_csv(file_path: str) -> np.ndarray:
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Extract the pulse values
    pulse_values = df[' PULSE'].values
    
    # Normalize and clip the pulse values
    pulse_values = pulse_values / 90.0
    #normalize_and_clip(pulse_values)
    
    return pulse_values

# Splits the longer recordings into segments as you need it 
def split_in_segments(signal: np.ndarray, pulse: np.ndarray, seg_len_seconds: int, fa_desired: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a whole measurement into different segments and down-sample data.

    :param signal: Array containing the PPG signal
    :param pulse: Array containing the pulse values
    :param seg_len_seconds: Length of extracted segments in seconds
    :param fa_desired: Sampling rate after down-sampling
    :return: Segmented data[samples, segments], pulse values[segments]
    """
    fa_cur = 125  # Original sampling rate
    N_samples = signal.shape[0]  # Samples in the signal
    N_samples_seg_cur = seg_len_seconds * fa_cur  # Samples in current segment
    N_samples_seg_new = seg_len_seconds * fa_desired  # Samples in segment after
    
    # How many splits and where to split
    N_seg = N_samples // N_samples_seg_cur
    split_idx = np.asarray(range(1, N_seg)) * N_samples_seg_cur

    # Initialize new array
    new_segments = np.zeros((N_samples_seg_new, N_seg))
    pulse_segments = np.zeros(N_seg)
    last_split = 0

    # Split array
    for i, split in enumerate(split_idx):
        cur_seg = signal[last_split:split]
        # Perform linear interpolation
        x_old = np.linspace(0, 1, len(cur_seg))
        x_new = np.linspace(0, 1, fa_desired * seg_len_seconds)
        new_segments[:, i] = np.interp(x_new, x_old, cur_seg)
        pulse_segments[i] = pulse[i]  # Assuming pulse is sampled at 1 Hz and corresponds to each segment

        # Prepare for the next split
        last_split = split

    # Normalize the data
    mean = np.mean(new_segments, axis=0, keepdims=True)
    std_dev = np.std(new_segments, axis=0, keepdims=True)

    # Replace zero std dev with 1 to avoid division by zero
    std_dev[std_dev == 0] = 1

    new_segments = (new_segments - mean) / std_dev

    return new_segments, pulse_segments

# Get the path to your downloaded dataset and returns it in a segmented manner     
def get_segments(path_dir: Path, sampling_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    all_data = []
    all_pulse = []

    path_all_signal_files = path_dir / "*_Signals.csv"
    path_all_pulse_files = path_dir / "*_Numerics.csv"
    path_all_signal_files = glob.glob(str(path_all_signal_files.absolute()))
    path_all_pulse_files = glob.glob(str(path_all_pulse_files.absolute()))
    path_all_signal_files = sorted(path_all_signal_files)
    path_all_pulse_files = sorted(path_all_pulse_files)

    print(f"Found {len(path_all_signal_files)} signal files to process.")
    print(f"Found {len(path_all_pulse_files)} pulse files to process.")

    for signal_file, pulse_file in zip(path_all_signal_files, path_all_pulse_files):
        try:
            signal = extract_signal_from_csv(signal_file)
            pulse = extract_pulse_from_csv(pulse_file)
            print(f"Processing files: {signal_file} and {pulse_file}")
            seg_len = 7  # 7 seconds of data
            data, pulse_segments = split_in_segments(signal, pulse, seg_len_seconds=seg_len, fa_desired=sampling_rate)

            if data.size > 0:
                all_data.append(data)
                all_pulse.append(pulse_segments)
                print(f"Segments extracted from {signal_file}: {data.shape[1]} valid segments.")
            else:
                print(f"No valid segments found in file: {signal_file}")

        except ValueError as e:
            print(f"Problem reading file: {signal_file}, error: {e}")
            continue

    if not all_data:
        raise ValueError("No data was collected; please check the input files.")

    return np.concatenate(all_data, axis=1), np.concatenate(all_pulse)

############################################## Main Part of the Script #######################
if __name__ == "__main__":
    # Check if GPUs are available
#    gpus = tf.config.experimental.list_physical_devices('GPU')
#    print("GPUs Available: ", gpus)
#    if gpus:
#        try:
#            for gpu in gpus:
#                tf.config.experimental.set_memory_growth(gpu, True)
#        except RuntimeError as e:
#            print(e)  # Memory growth must be set at program startup

    # Get the database 
    # This download step takes huge amounts of time. Only do this once, then comment it out again
    os.system('wget -r -N -c -np https://physionet.org/files/bidmc/1.0.0/')
    
    # After the download, your data should be in these directories
    path_bidmc = Path("physionet.org/files/bidmc/1.0.0/bidmc_csv/")

    # Dividing in test and training data 
    x_data, y_pulse = get_segments(path_bidmc, sampling_rate=25)

    # Ensure no NaN or infinite values
    x_data = np.nan_to_num(x_data)
    y_pulse = np.nan_to_num(y_pulse)

    # Print shapes to verify
    print(f"x_data shape: {x_data.shape}")  # Should be (samples, segments)
    print(f"y_pulse shape: {y_pulse.shape}")  # Should be (segments,)

    # Dividing in test and training data 
    num_segments = x_data.shape[1]  # Number of segments
        
    train_size = int(0.7 * num_segments)
    val_size = int(0.1 * num_segments)
    x_train = x_data[:, :train_size]  # (samples, train_size)
    y_train = y_pulse[:train_size]  # (train_size,)

    x_val = x_data[:, train_size:train_size + val_size] 
    y_val = y_pulse[train_size:train_size + val_size]

    x_test = x_data[:, train_size + val_size:]     # (samples, num_segments - train_size)
    y_test = y_pulse[train_size + val_size:]       # (num_segments - train_size,)
        
    # Shuffle only the training data
    train_indices = np.arange(train_size)
    np.random.seed(51)  # Set seed for shuffling
    np.random.shuffle(train_indices)
    # Shuffle the training data
    x_train = x_train[:, train_indices]
    y_train = y_train[train_indices]
            
    # Reshape the input data
    x_train = x_train.transpose(1, 0)  # Shape becomes (train_size, samples)
    x_test = x_test.transpose(1, 0)    # Shape becomes (test_size, samples)
    x_val = x_val.transpose(1, 0)      # Shape becomes (val_size, samples)

    # Print shapes after transposing
    print(f"x_train shape after transpose: {x_train.shape}")  # Should be (train_size, samples)
    print(f"x_test shape after transpose: {x_test.shape}")    # Should be (test_size, samples)
    print(f"x_val shape after transpose: {x_val.shape}")      # Should be (val_size, samples)
    
    # After selecting only the first 75 data points for each sample
    x_train = x_train[:, :75]  # Shape becomes (2522, 75)
    x_test = x_test[:, :75]    # Shape becomes (722, 75)
    x_val = x_val[:, :75]      # Shape becomes (360, 75)

    # Print shapes after selecting 75 data points
    print(f"x_train shape after selecting 75 data points: {x_train.shape}")  # Should be (2522, 75)
    print(f"x_test shape after selecting 75 data points: {x_test.shape}")    # Should be (722, 75)
    print(f"x_val shape after selecting 75 data points: {x_val.shape}")      # Should be (360, 75)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)  # Shape: (2522, 75)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Shape: (2522, 1)
    x_val = torch.tensor(x_val, dtype=torch.float32)      # Shape: (360, 75)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)      # Shape: (360, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)    # Shape: (722, 75)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)    # Shape: (722, 1)

    # Add channel dimension correctly
    x_train = x_train.unsqueeze(1)  # Shape becomes (2522, 1, 75)
    x_val = x_val.unsqueeze(1)      # Shape becomes (360, 1, 75)
    x_test = x_test.unsqueeze(1)    # Shape becomes (722, 1, 75)

    # Print shapes after adding channel dimension
    print(f"x_train shape after adding channel dimension: {x_train.shape}")  # Should be (2522, 1, 75)
    print(f"x_val shape after adding channel dimension: {x_val.shape}")      # Should be (360, 1, 75)
    print(f"x_test shape after adding channel dimension: {x_test.shape}")    # Should be (722, 1, 75)
    # Create datasets
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    # Create DataLoaders with correct batch size
    train_loader = DataLoader(train_dataset, batch_size=150, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=150, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False)

    # Check the shape of one batch
    for x_batch, y_batch in train_loader:
        print(f"x_batch shape: {x_batch.shape}")  # Should print [150, 1, 75]
        break  # Just check one batch to verify
    
    # Print shapes to verify
    print(f"x_train shape after selecting samples: {x_train.shape}")
    print(f"x_test shape after selecting samples: {x_test.shape}")

    # Verify the sliced data 
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    
    # Define the PyTorch model
    class PPGModel(nn.Module):
        def __init__(self):
            super(PPGModel, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=8, padding=0)
            self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8, padding=0)
            self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, padding=0)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(32, 16)
            self.fc2 = nn.Linear(16, 1)  # Regression output

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.gap(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    
    # Initialize the model, loss function, and optimizer
    model = PPGModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.007)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # Pruning
    def prune_model(model, amount=0.90):
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        for module, param in parameters_to_prune:
            prune.remove(module, param)  # This line ensures the mask is applied
        return model

    model_prune = prune_model(model, amount=0.90)

    # Fine-tuning after pruning
    optimizer = optim.Adam(model_prune.parameters(), lr=0.0005)
    for epoch in range(50):
        model_prune.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model_prune(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model_prune.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model_prune(x_batch)
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{50}, Validation Loss: {val_loss:.4f}")

    # Evaluation
    model_prune.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = model_prune(x_batch)
            y_pred_list.append(y_pred.numpy())
            y_true_list.append(y_batch.numpy())

    # Scale it back to the original
    y_pred = np.concatenate(y_pred_list) * 90
    y_true = np.concatenate(y_true_list) * 90 
    
    # Print some examples of predictions and true labels
    n_examples = 15  # Number of examples to print
    print("\nExample Predictions vs. True Labels:")
    for i in range(n_examples):
        print(f"Predicted: {y_pred[i][0]:.4f}, True Label: {y_true[i][0]:.4f}")

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE:  {mse:.4f}")
    print(f"R2:   {r2:.4f}")
    

    print("\n Quantizing Model .... ") 

    # Apply dynamic quantization on the trained model
    quantized_model = torch.quantization.quantize_dynamic(model_prune, {nn.Linear, nn.Conv1d}, dtype=torch.qint8)

    # Now you can run inference on the quantized model
    quantized_model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred = quantized_model(x_batch)
            y_pred_list.append(y_pred.numpy())
            y_true_list.append(y_batch.numpy())

    # Scale it back to the original
    y_pred = np.concatenate(y_pred_list) * 90
    y_true = np.concatenate(y_true_list) * 90 

    # Print some examples of predictions and true labels
    n_examples = 15  # Number of examples to print
    print("Example Predictions vs. True Labels:")
    for i in range(n_examples):
        print(f"Predicted: {y_pred[i][0]:.4f}, True Label: {y_true[i][0]:.4f}")
        
    # Calculate metrics for the quantized model
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Quantized Model MSE:  {mse:.4f}")
    print(f"Quantized Model R2:   {r2:.4f}")
    
    model_json = convert_model_to_json(model_prune, x_train.shape)
    
    with open('./output/model_arch/modelTest.json', 'w') as f:
        json.dump(model_json, f)
        
        
    with open('./output/model_arch/modelTest.json') as f:
        json_data = json.load(f)
    json_str = json.loads(json_data)

    # This bitwidth should correspond to the previously tested quantization 
    BIT_WIDTH = 8
    INT_WIDTH = 3
    FRAC_WIDTH = BIT_WIDTH - INT_WIDTH

    # Output directory of the accelerator 
    output_dir = './output/test/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the weights
    #tinyhls.extract_weights(model_prune, './output/test/weights/')
    extract_weights_torch(model_prune, './output/test/weights/')
    
    path_temp = os.getcwd()
    source_dir = os.path.join(path_temp, 'output/test/weights')   # replace this with the path to your directory
    pattern = r'([wb])(\d+)\.txt'
    files = os.listdir(source_dir)
    txt_files = [f for f in files if f.endswith('.txt')]
    hex_files = [f for f in files if f.endswith('.hex')]

    txt_files.sort(key=lambda x: int(re.search(pattern, x).group(2)))
    hex_files.sort()
    quantization = {'total': BIT_WIDTH, 'int': INT_WIDTH, 'frac': FRAC_WIDTH}

    tinyhls.convert_weights_to_hex(source_dir, source_dir, txt_files, BIT_WIDTH, INT_WIDTH)
    tinyhls.convert_bias_to_hex(source_dir, source_dir, txt_files, BIT_WIDTH, INT_WIDTH)
    tinyhls.create_verilog_includes(source_dir, source_dir, json_str, BIT_WIDTH)
    tinyhls.translate_model(model_arch=json_str, param_path=source_dir, output_path=output_dir, fast=False, quantization=quantization, file_name="tinyhls_cnn_test")
    tinyhls.create_testbench(model_arch=json_str, quantization=quantization, clk_period=100, destination_path=output_dir, file_name='tinyhls_tb_test')

    print("End of test.py; Translation done!")
    
    
    
    

