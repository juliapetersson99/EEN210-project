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
from sklearn.preprocessing import StandardScaler

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



def segment_file(df, sensor_columns, window_size, overlap, format='values', labeling_mode = 'mode'):    

    step = int(window_size * (1 - overlap))
    file_windows = []
    file_labels = []


    if(format == 'values'):
        data_matrix = df[sensor_columns].values  # shape: (num_samples, num_channels)
        
        # Slide over the data in steps, discarding any incomplete tail
        for start in range(0, len(df) - window_size + 1, step):
            end = start + window_size
            segment = data_matrix[start:end, :]  # shape: (window_size, num_channels)
            
            if(labeling_mode == 'mode'):
                window_labels = df['label'].iloc[start:end]
                label_window = window_labels.mode()[0]
            else:
                window_labels = df['label'].iloc[start:end]
                label_window = window_labels.iloc[-1]
                #print(f'window_labels: {window_labels}')
                #print(f'label: {label_window}')


     

            
            file_windows.append(segment)
            file_labels.append(label_window)


        
        return file_windows, file_labels
    else:
        data_matrix = df[sensor_columns]  # shape: (num_samples, num_channels)
        #print(data_matrix)
        # Slide over the data in steps, discarding any incomplete tail
        for start in range(0, len(df) - window_size + 1, step):
            end = start + window_size
            segment = data_matrix.iloc[start:end] # shape: (window_size, num_channels)
            
            # Determine the mode label in this window

                        # Determine the mode label in this window
            if(labeling_mode == 'mode'):
                window_labels = df['label'].iloc[start:end]
                label_window = window_labels.mode()[0]
            else:
                window_labels = df['label'].iloc[start:end]
                label_window = window_labels.iloc[-1]
                #print(f'window_labels: {window_labels}')
                #print(f'label: {label_window}')
            
            file_windows.append(segment)
            file_labels.append(label_window)


        
        return file_windows, file_labels




def add_features(data_frame, rolling_size):

    #add last two data points per window aswell
    data_types = ['acceleration', 'gyroscope']
    dimensions = ['x', 'y', 'z']
    output = dict()
    columns = set(data_frame.columns)
    for data_type in data_types:
        for dimension in dimensions:
            column_name = f"{data_type}_{dimension}"
            data = data_frame[column_name]

            data_frame[f"{column_name}_mean"] = data.rolling(window=rolling_size).mean()
            data_frame[f"{column_name}_max"] = data.rolling(window=rolling_size).max()
            data_frame[f"{column_name}_min"] = data.rolling(window=rolling_size).min()
            data_frame[f"{column_name}_std"] = data.rolling(window=rolling_size).std()
            data_frame[f"{column_name}_median"] = data.rolling(window=rolling_size).median()

        data_frame[f"{data_type}_magnitude"] = np.sqrt(
            data_frame[f"{data_type}_x"] ** 2 +
            data_frame[f"{data_type}_y"] ** 2 +
            data_frame[f"{data_type}_z"] ** 2
        )
        # output[f"{data_type}_magnitude_std"] = data_frame[f"{data_type}_magnitude"].std()
    addedColumns = set(data_frame.columns) - columns
    return data_frame, list(addedColumns)


def get_window_statistics(data_windos):
    data_types = ['acceleration', 'gyroscope']
    dimensions = ['x', 'y', 'z']
    output = dict()
    for data_type in data_types:
        for dimension in dimensions:
            column_name = f"{data_type}_{dimension}"
            data = data_windos[column_name]

            output[f"{column_name}_mean"] = data.mean()
            output[f"{column_name}_max"] = data.max()   
            output[f"{column_name}_min"] = data.min()
            output[f"{column_name}_std"] = data.std()
            output[f"{column_name}_median"] = data.median()
            #output[f"{column_name}_kurtosis"] = data.kurtosis()
            #output[f"{column_name}_skewness"] = data.skew()
            #for x in range(1,3):
             #   output[f"{column_name}_last_{str(x)}"] = data.iloc[-x] 
        magnitude = np.sqrt(
            data_windos[f"{data_type}_x"] ** 2 +
            data_windos[f"{data_type}_y"] ** 2 +
            data_windos[f"{data_type}_z"] ** 2
        )
        output[f"{data_type}_magnitude"] = magnitude.mean()
        output[f"{data_type}_magnitude_std"] = magnitude.std()

    
    return output


def load_and_preprocess_data(data_folder, window_duration_sec=1.5, fs=60, overlap=0.5):
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    sensor_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                      'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    window_size = int(window_duration_sec * fs)
    all_windows = []
    all_labels = []

    dfs = [pd.read_csv(file) for file in csv_files]

    scaler = StandardScaler()
    scaler.fit(pd.concat(dfs)[sensor_columns])
    for file, df in zip(csv_files, dfs):
        print(f"Loading {file}")
        df = df.copy()# warnings otherwise to allocate to the for-loop coopy obhect

        # Sort by timestamp if needed
        df = add_missing_labels(df)
        df = df[df['label'] != 'none']

        df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'])

        df[sensor_columns] = scaler.transform(df[sensor_columns])
    

        df, statistic_columns = add_features(df, window_size)
        df = df[window_size-1:]

        time_diffs = df['timestamp'].diff()

        # Calculate the average time difference in seconds
        average_time_diff_seconds = time_diffs.dt.total_seconds().mean()

        # Calculate the frequency
        frequency = 1 / average_time_diff_seconds

        print(f"The average frequency is approximately {frequency:.2f} Hz")

        print(f"Loaded {len(df)} samples")
        # low_pass_filter(df[sensor_columns], cutoff=3, fs=fs, order=3)
        # print("Applied low-pass filter")
        

        # Segment this file into windows
        file_windows, file_labels = segment_file(
            df, sensor_columns + statistic_columns, window_size, overlap
        )
        print(f"Segmented into {len(file_windows)} windows")

        
        all_windows.extend(file_windows)
        all_labels.extend(file_labels)
    X = np.array(all_windows)  # shape: (total_segments, window_size, num_channels)
    y = np.array(all_labels)

    return X, y

        
 
def load_and_preprocess_static_picture(data_folder, window_duration_sec=1.5, fs=60, overlap=0.5, labeling_mode = 'mode'):
    csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
    sensor_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z',
                      'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    window_size = int(window_duration_sec * fs)
    all_windows = []
    all_labels = []

    dfs = [pd.read_csv(file) for file in csv_files]

    for file, df in zip(csv_files, dfs):
        print(f"Loading {file}")
        df = df.copy()# warnings otherwise to allocate to the for-loop coopy obhect

        # Sort by timestamp if needed
        df = add_missing_labels(df)
        df = df[df['label'] != 'none']

        df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'])

        time_diffs = df['timestamp'].diff()

        # Calculate the average time difference in seconds
        average_time_diff_seconds = time_diffs.dt.total_seconds().mean()

        # Calculate the frequency
        frequency = 1 / average_time_diff_seconds

        print(f"The average frequency is approximately {frequency:.2f} Hz")

        print(f"Loaded {len(df)} samples")
        # low_pass_filter(df[sensor_columns], cutoff=3, fs=fs, order=3)
        # print("Applied low-pass filter")
        
        # Segment this file into windows
        file_windows, file_labels = segment_file(
            df, sensor_columns, window_size, overlap, format='dataframe', labeling_mode = labeling_mode
        )
        print(f"Segmented into {len(file_windows)} windows")
        for window in file_windows:
            window_stats = get_window_statistics(window)
            all_windows.append(window_stats)

        all_labels.extend(file_labels)
    #print(window_stats)
    keys = all_windows[0].keys()  # Assumes all dicts have the same keys.
    data = [[d[k] for k in keys] for d in all_windows]

    X = np.array(data)  # shape: (total_segments,  num_channels)
    y = np.array(all_labels)

    return X, y

        
 
    


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

