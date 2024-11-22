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
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
#from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

import json
import re
import tinyhls
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import random
import logging
from datetime import datetime

#Pruning
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning, ConstantSparsity, prune_low_magnitude
#import tensorflow_model_optimization.sparsity as sparsity
import tempfile

# Quantization 
from fxpmath import Fxp

# Provided from Felix Wichum 
import wfdb
from pathlib import Path
from typing import Tuple
import glob
from scipy import signal

import time
from tqdm import tqdm


######################## Helper Functions ##############################

# To verify that your path to the data actually contains data 
def list_files_recursively(path):
    for file in Path(path).rglob('*'):
        print(file)

# Extracts the signals from the database files 
def extract_signal_from_file(file_path: str) -> Tuple[wfdb.Record, wfdb.Annotation]:
    # get whole record
    record = wfdb.rdrecord(file_path, sampfrom=0, sampto=None)
    # get annotations
    annotation = wfdb.rdann(file_path, 'atr', sampfrom=0, sampto=None)
    return record, annotation

# Splits the longer recordings into segments as you need it 
def split_in_segments(rec: wfdb.Record, ann: wfdb.Annotation, seg_len_seconds: int, fa_desired: int, is_nsrdb: bool = False, channels: Tuple[int, int] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a whole measurement into different segments. Get label for each segment and down-sample data.

    :param rec: Recording of the data
    :param ann: Annotation of the data
    :param seg_len_seconds: Length of extracted segments in seconds
    :param fa_desired: Sampling rate after down-sampling
    :param is_nsrdb: Boolean indicating if the data is from NSRDB (no AFIB labels)
    :param channels: Tuple indicating which two channels to use
    :return: Segmented data[channel, samples, segments], labels[segments]
    """
    # Get current data
    fa_cur = ann.fs
    N_samples = rec.sig_len  # Samples in the signal
    N_samples_seg_cur = seg_len_seconds * fa_cur  # Samples in current segment
    N_samples_seg_new = seg_len_seconds * fa_desired  # Samples in segment after
    rhythm_lab = np.asarray(ann.aux_note)
    rhythm_lab_sample = ann.sample

    # How many splits and where to split
    N_seg = N_samples // N_samples_seg_cur
    split_idx = np.asarray(range(1, N_seg)) * N_samples_seg_cur

    # Initialize new arrays
    new_channels = np.zeros((2, N_samples_seg_new, N_seg))  # Only use two channels
    labels = np.zeros((N_seg))
    last_split = 0
    last_afib = False
    # Initialize counters
    positive_count = 0
    negative_count = 0

    # Split array
    for i, split in enumerate(split_idx):
        # Get data for the specified channels
        for idx, ch in enumerate(channels):
            cur_ch = rec.p_signal[last_split:split, ch]
            new_channels[idx, :, i] = signal.resample(cur_ch, fa_desired * seg_len_seconds)

        # Get label
        if is_nsrdb:
            # For NSRDB, set label to 0 (no AFIB)
            labels[i] = 0
            negative_count += 1
        else:
            # For AFDB, check annotations
            label_index = np.where(np.logical_and(rhythm_lab_sample >= last_split, rhythm_lab_sample < split))[0].astype(int)
            current_annotations = rhythm_lab[label_index]
            current_afib = False
            
            # Set label to AFIB (True) if any AFIB occurred in this segment
            if '(AFIB' in current_annotations:
                current_afib = True

            labels[i] = last_afib or current_afib
            if len(current_annotations) >= 1 and current_annotations[-1] == '(AFIB':
                last_afib = True
                positive_count += 1
            elif len(current_annotations) == 0:
                last_afib = last_afib  # keep the old one
                positive_count += 1
            else:
                last_afib = False
                negative_count += 1

        # Prepare for the next split
        last_split = split

    # Print counts of positive and negative samples
    print(f"Positive samples (AFIB): {positive_count}, Negative samples (N): {negative_count}")

    # Filter out segments without valid annotations
    valid_segments = np.where(np.isin(labels, [0, 1]))[0]
    new_channels = new_channels[:, :, valid_segments]
    labels = labels[valid_segments]

    # Normalize the data
    mean = np.mean(new_channels, axis=1, keepdims=True)
    std_dev = np.std(new_channels, axis=1, keepdims=True)

    # Replace zero std dev with 1 to avoid division by zero
    std_dev[std_dev == 0] = 1

    new_channels = (new_channels - mean) / std_dev

    return new_channels, labels

# in the MIT-BIH Atrial Fibrillation Database, there are some signals that contain 
# Quote:  isolated data blocks from the original tapes were unreadable. In these cases, the missing data, corresponding to 10.24 seconds for each missing block, have been replaced with a flat segment of samples with amplitudes of zero.
# Here these segments are skipped for the dataset
def is_segment_valid(segment: np.ndarray) -> bool:
    threshold = segment.size // 2  # Half the number of samples in the segment
    return np.count_nonzero(segment) > threshold  # Check if more than half are non-zero
    
# Get the path to your downloaded dataset and returns it in a segmented and labeled manner     
def get_segments(path_dir: Path, sampling_rate: int, is_nsrdb: bool = False, channels: Tuple[int, int] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    all_data = []
    all_labels = []

    path_all_files = path_dir / "*.hea"
    path_all_files = glob.glob(str(path_all_files.absolute()))
    path_all_files = sorted(path_all_files)
    path_all_files = [f[:-4] for f in path_all_files]

    print(f"Found {len(path_all_files)} files to process.")

    for file in path_all_files:
        try:
            rec, ann = extract_signal_from_file(file)
            print(f"Processing file: {file}")
            seg_len = int(800 / sampling_rate)
            data, label = split_in_segments(rec, ann, seg_len_seconds=seg_len, fa_desired=sampling_rate, is_nsrdb=is_nsrdb, channels=channels)

            # Filter out invalid segments
            valid_segments = [i for i in range(data.shape[2]) if is_segment_valid(data[:, :, i])]
            data = data[:, :, valid_segments]
            label = label[valid_segments]

            if data.size > 0 and label.size > 0:
                all_data.append(data)
                all_labels.append(label)
                print(f"Segments extracted from {file}: {data.shape[2]} valid segments.")
            else:
                print(f"No valid segments found in file: {file}")

        except ValueError as e:
            print(f"Problem reading file: {file}, error: {e}")
            continue

    if not all_data or not all_labels:
        raise ValueError("No data was collected; please check the input files.")

    all_data = np.concatenate(all_data, axis=2)
    all_labels = np.concatenate(all_labels)

    # Filter to keep only positive samples for AFDB and negative samples for NSRDB
    if is_nsrdb:
        filtered_indices = np.where(all_labels == 0)[0]
    else:
        filtered_indices = np.where(all_labels == 1)[0]

    all_data = all_data[:, :, filtered_indices]
    all_labels = all_labels[filtered_indices]

    return all_data, all_labels

# Function to create a new model with ReLU activation in the final layer
def create_model_with_relu():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', input_shape=(512, 2), kernel_initializer='he_uniform', name='conv1d'))
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', kernel_initializer='he_uniform', name='conv1d_1'))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu', kernel_initializer='he_uniform', name='conv1d_2'))
    model.add(tf.keras.layers.GlobalAveragePooling1D(name='global_average_pooling1d'))
    model.add(tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='he_uniform', name='dense'))
    model.add(tf.keras.layers.Dense(units=2, activation='relu', kernel_initializer='he_uniform', name='dense_1'))  # Changed activation to ReLU
    return model

############################################## Main Part of the Script #######################
if __name__ == "__main__":
    # Check if GPUs are available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("GPUs Available: ", gpus)
    if gpus:
        try:
            os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)  # Memory growth must be set at program startup
     
    # Get the database 
    # This download step takes huge amounts of time. Only do this once, then comment it out again
    os.system('wget -r -N -c -np https://physionet.org/files/afdb/1.0.0/')
    os.system("wget -r -N -c -np https://physionet.org/files/nsrdb/1.0.0/")
    
    # After the download, your data should be in these directories
    path_nsrdb = Path("physionet.org/files/nsrdb/1.0.0/")
    path_afdb = Path("physionet.org/files/afdb/1.0.0/")

    # Get segments from the databases
    channels_to_use = (0, 1)  # Specify which channels to use
    x_data_afdb, y_data_afdb = get_segments(path_afdb, sampling_rate=256, channels=channels_to_use)
    x_data_nsrdb, y_data_nsrdb = get_segments(path_nsrdb, is_nsrdb=True, sampling_rate=256, channels=channels_to_use)
    
    # Convert data into pandas DataFrames
    data_si = pd.DataFrame({'ch1': list(x_data_afdb[0]), 'ch2': list(x_data_afdb[1])})
    data_si['class'] = 'si'
    data_af = pd.DataFrame({'ch1': list(x_data_nsrdb[0]), 'ch2': list(x_data_nsrdb[1])})
    data_af['class'] = 'af'

    print("Data SI:")
    print(data_si.head())
    print("Data AF:")
    print(data_af.head())

    data = pd.concat([data_si, data_af], axis=0)
    
    le = LabelEncoder()
    labels_numerical = le.fit_transform(data['class'])
    print(data.shape)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(['class'], axis=1), labels_numerical, test_size=0.2, random_state=42, stratify=labels_numerical)
    
    # Process the data arrays directly
    data_si_full = [x_data_afdb[0], x_data_afdb[1]]
    data_af_full = [x_data_nsrdb[0], x_data_nsrdb[1]]
    
    # Keep only last 2048 elements
    data_si_purge = [[], []]
    for i in range(len(data_si_full[0])):
        data_si_purge[0].append(data_si_full[0][i][-2048:])
        data_si_purge[1].append(data_si_full[1][i][-2048:])

    data_af_purge = [[], []]
    for i in range(len(data_af_full[0])):
        data_af_purge[0].append(data_af_full[0][i][-2048:])
        data_af_purge[1].append(data_af_full[1][i][-2048:])
        
    # Subsample by factor of 4
    data_si_subsample = [[], []]
    for i in range(len(data_si_purge[0])):
        data_si_subsample[0].append(data_si_purge[0][i][::2])
        data_si_subsample[1].append(data_si_purge[1][i][::2])

    data_af_subsample = [[], []]
    for i in range(len(data_af_purge[0])):
        data_af_subsample[0].append(data_af_purge[0][i][::2])
        data_af_subsample[1].append(data_af_purge[1][i][::2])

    # Convert to numpy arrays for further processing
    data_si_subsample = np.array(data_si_subsample)
    data_af_subsample = np.array(data_af_subsample)

    # Standardize the data
    scaler = StandardScaler()
    data_si_subsample = scaler.fit_transform(data_si_subsample.reshape(-1, data_si_subsample.shape[-1])).reshape(data_si_subsample.shape)
    data_af_subsample = scaler.transform(data_af_subsample.reshape(-1, data_af_subsample.shape[-1])).reshape(data_af_subsample.shape)

    print("Data SI Subsampled and Standardized:")
    print(data_si_subsample)
    print("Data AF Subsampled and Standardized:")
    print(data_af_subsample)

    # Prepare data for model training
    data_si_np = np.ones(shape=(len(data_si_subsample[0]), 512, 2), dtype=np.int16)
    data_af_np = np.ones(shape=(len(data_af_subsample[0]), 512, 2), dtype=np.int16)
    
    for ch in range(2):
        for row in range(len(data_si_subsample[0])):
            for val in range(512):
                try:
                    data_si_np[row, val, ch] = data_si_subsample[ch][row][val]
                except IndexError:
                    print(ch, row, val)

    for ch in range(2):
        for row in range(len(data_af_subsample[0])):
            for val in range(512):
                try:
                    data_af_np[row, val, ch] = data_af_subsample[ch][row][val]
                except IndexError:
                    print(ch, row, val)
                    
    data_af_np.shape
    nn_data = np.vstack((data_si_np, data_af_np))
    nn_data.shape
    af_labels = [1]*len(data_af_subsample[0])
    si_labels = [0]*len(data_si_subsample[0])
    class_labels = si_labels + af_labels
    len(class_labels)
    
    class_labels_encoded = tf.keras.utils.to_categorical(class_labels, 2, "int")
    
    data_af_np_withclass = np.ones((len(data_af_subsample[0]), 512, 3), dtype=np.uint16)
    data_si_np_withclass = np.ones((len(data_si_subsample[0]), 512, 3), dtype=np.uint16)

    data_af_np_withclass[:,:,0:2] = data_af_np
    data_si_np_withclass[:,:,0:2] = data_si_np

    af_labels = [1]*len(data_af_subsample[0])
    si_labels = [0]*len(data_si_subsample[0])

    data_af_np_withclass[:,:,2] = np.array(af_labels).reshape((len(data_af_subsample[0]),1))
    data_si_np_withclass[:,:,2] = np.array(si_labels).reshape((len(data_si_subsample[0]),1))
    
    data_af_np_withclass[(len(data_af_subsample[0])-1), 456]
    nn_data_withclass = np.vstack((data_si_np_withclass, data_af_np_withclass))
    nn_data_withclass[1535, 54]
    
    max_val = np.amax(nn_data)
    min_val = np.amin(nn_data)
    
    nn_data_signed = nn_data
    nn_data_signed = ((nn_data_signed - min_val) / (max_val - min_val)) * 255 - 127
    nn_data_signed = np.round(nn_data_signed).astype(np.int8)
    
    max_val_signed = np.amax(nn_data_signed)
    min_val_signed = np.amin(nn_data_signed)
    
    h, b = np.histogram(nn_data_signed, bins=256, density=True)
    
    sum = 0
    for i in range(120,136):
        sum = sum + h[i]
    print(sum)
    
    nn_data_signed_4bit = np.clip(nn_data_signed, -8, 7)
    nn_data_signed_4bit.shape
    
    #np.save('../Data/train_data/combined_data_signed_4bit.npy', nn_data_signed_4bit)
     
    #X_train, X_test, y_train, y_test = train_test_split(nn_data_signed_4bit, class_labels_encoded, test_size=0.2, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(nn_data_signed_4bit, class_labels_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    #Sanity Checks
    if X_train.size == 0:
        print('Error: X_train is empty')
    if X_test.size == 0:
        print('Error: X_test is empty')
    print(X_train.shape)
    print(data_si.iloc[0]['ch1'].size)
    
    # Basic model definition
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', input_shape=(512,2), kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(units=2, activation='sigmoid', kernel_initializer='he_uniform'))
    model.summary()

    # Training of the model
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
    history1 = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test), batch_size=10, verbose=2, shuffle = True)

    # Pruning by 75%
    pruning_params = {"pruning_schedule": ConstantSparsity(0.75, begin_step=0, frequency=100)}
    model_prune = prune_low_magnitude(model, **pruning_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_prune.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Rebuild the model after pruning
    model_prune = strip_pruning(model_prune)
    model_prune.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    logdir = tempfile.mkdtemp()

    callbacks = [
        pruning_callbacks.UpdatePruningStep(),
        pruning_callbacks.PruningSummaries(log_dir=logdir)
    ]
        
    history = model_prune.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size = 10, callbacks=callbacks, verbose=2, shuffle = True)
    
    model_prune = strip_pruning(model_prune)
    model_prune.summary()
    
    model_prune.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Recall()])
    test_loss, test_acc, test_recall = model_prune.evaluate(X_test,  y_test)
    
    # Make predictions on the validation dataset
    y_val_pred_prob = model_prune.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    y_val_true = np.argmax(y_val, axis=1)

    # Calculate F1-Score
    f1 = f1_score(y_val_true, y_val_pred)
    print(f"F1-Score: {f1:.4f}")

    # Calculate Accuracy
    accuracy = accuracy_score(y_val_true, y_val_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate Sensitivity (Recall)
    sensitivity = recall_score(y_val_true, y_val_pred)
    print(f"Sensitivity (Recall): {sensitivity:.4f}")

    # Calculate Specificity
    tn, fp, fn, tp = confusion_matrix(y_val_true, y_val_pred).ravel()
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity:.4f}")



######### Hardware Specific Things #################
    #Exchange Sigmoid with ReLU 
    #Sigmoid brings unneccessary hardware overhead, 
    #since it just selects the larger output anyways. 
    
    #Check current model architecture
    model_prune.summary()
    # Create a new model with ReLU activation in the final layer
    # This had to be outsourced in a function to ensure consistent numbering
    new_model = create_model_with_relu()
    #Verify new model structure 
    new_model.summary()
    # Transfer weights from the original model to the new model
    for layer in model_prune.layers[:-1]:  # Skip the last layer
        new_model.get_layer(layer.name).set_weights(layer.get_weights())

    # Compile the new model
    new_model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])

    # Export model to JSON
    model_json = new_model.to_json()

    with open('./output/model_arch/modelTest.json', 'w') as f:
        json.dump(model_json, f)

    with open('./output/model_arch/modelTest.json') as f:
        json_data = json.load(f)
    json_str = json.loads(json_data)
    
    
    ind = random.randint(0,len(X_test))
    x = X_test[ind]
    x = np.expand_dims(x, axis=0)
    x.shape

    x = X_test[ind]
    x = np.expand_dims(x, axis=0)
    x.shape

    # This bitwidth should correspond to the previously tested quantization 
    BIT_WIDTH = 8
    INT_WIDTH = 4
    FRAC_WIDTH = BIT_WIDTH - INT_WIDTH

    # Output directory of the accelerator 
    output_dir = './output/test/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the weights
    tinyhls.extract_weights(model_prune, './output/test/weights/')
    
    source_dir = './output/test/weights/'
    files = os.listdir(source_dir)
    
    # Verifying the content of the weights 
    #for file in files:
    #    with open(os.path.join(source_dir, file), 'r') as f:
    #        content = f.read()
    #        print(f"Content of {file}: {content}")


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

    print("End of ecg_example.py; Translation done!")
    
    
    
    

