import pandas as pd
from scipy.signal import butter, filtfilt
import glob
import os
import numpy as np


##
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def split_dataframe_into_sliding_windows(df, window_size):
 
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    windowed_dfs = []
    start = df.index.min()
    end = df.index.max()

    current_start = start
    while current_start < end:
        #print(current_start)
        current_end = current_start + pd.Timedelta(window_size)
        window = df[current_start:current_end]
        if not window.empty:
            windowed_dfs.append(window.reset_index())
        current_start += pd.Timedelta(window_size)

    return windowed_dfs



def add_missing_labels(data_frame):
    # Replace deprecated fillna(method='ffill') with ffill()
    data_frame['label'] = data_frame['label'].ffill()
    return data_frame   




def low_pass_filter(data_frame, cutoff=3, fs=100, order=3):
    # Calculate the Nyquist frequency and the normalized cutoff
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    #get the filter to be applied
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    is_dataframe = isinstance(data_frame, pd.DataFrame)
    data = data_frame.values if is_dataframe else data_frame

    # Apply zero-phase filtering along axis 0 (filter each column independently).
    filtered_data = filtfilt(b, a, data, axis=0)
    if is_dataframe:
            filtered_data = pd.DataFrame(filtered_data, 
                                        index=data_frame.index, 
                                        columns=data_frame.columns)

    
    return filtered_data



def segment_file(df, sensor_columns, window_size, overlap):

    step = int(window_size * (1 - overlap))
    file_windows = []
    file_labels = []
    data_matrix = df[sensor_columns].values  # shape: (num_samples, num_channels)
    
    # Slide over the data in steps, discarding any incomplete tail
    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        segment = data_matrix[start:end, :]  # shape: (window_size, num_channels)
        
        # Determine the mode label in this window
        window_labels = df['label'].iloc[start:end]
        mode_label = window_labels.mode()[0]
        
        file_windows.append(segment)
        file_labels.append(mode_label)
    
    return file_windows, file_labels

def load_and_preprocess_data(data_folder, window_duration_sec=1.5, fs=60, overlap=0.5):
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    sensor_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                      'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    window_size = int(window_duration_sec * fs)
    all_windows = []
    all_labels = []
    for file in csv_files:
        print(f"Loading {file}")
        df = pd.read_csv(file)

        # Sort by timestamp if needed
        df = add_missing_labels(df)
        df = df[df['label'] != 'none']

        df['timestamp'] = pd.to_datetime(df['timestamp'])


        time_diffs = df['timestamp'].diff()

        # Calculate the average time difference in seconds
        average_time_diff_seconds = time_diffs.dt.total_seconds().mean()

        # Calculate the frequency
        frequency = 1 / average_time_diff_seconds

        print(f"The average frequency is approximately {frequency:.2f} Hz")

        print(f"Loaded {len(df)} samples")
        low_pass_filter(df[sensor_columns], cutoff=3, fs=fs, order=3)
        print("Applied low-pass filter")
        

        # Segment this file into windows
        file_windows, file_labels = segment_file(
            df, sensor_columns, window_size, overlap
        )
        print(f"Segmented into {len(file_windows)} windows")

        all_windows.extend(file_windows)
        all_labels.extend(file_labels)
        
    X = np.array(all_windows)  # shape: (total_segments, window_size, num_channels)
    # Min-max normalize each channel independently
    try:
        channel_min = X.min(axis=(0, 1), keepdims=True)  # shape: (1, 1, num_channels)
    except Exception as e:
        print(e)
        print(X)
    channel_max = X.max(axis=(0, 1), keepdims=True)  # shape: (1, 1, num_channels)
    channel_range = channel_max - channel_min
    channel_range[channel_range == 0] = 1e-8
    X_norm = (X - channel_min) / channel_range
    norm_min = X_norm.min(axis=(0, 1))
    norm_max = X_norm.max(axis=(0, 1))

    print("After normalization:")
    for idx, col in enumerate(sensor_columns):
        print(f"{col}: min = {norm_min[idx]:.4f}, max = {norm_max[idx]:.4f}")


    y = np.array(all_labels)
    return X_norm, y

        
    


def encode_labels(labels):
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_to_idx[label] for label in labels])
    return encoded, label_to_idx


def train_model(model, train_loader, test_loader, num_epochs=20, lr=0.001, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # expects raw data, not softmax
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        

        # Eval
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                test_loss += loss.item() * batch_data.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += batch_labels.size(0)
                correct_test += (predicted == batch_labels).sum().item()
        
        test_loss = test_loss / total_test
        test_acc = 100.0 * correct_test / total_test
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"| Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return model

