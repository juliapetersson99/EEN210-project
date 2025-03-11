"""Common functions and constants for the project."""

import pandas as pd
import numpy as np

SENSOR_COLS = [
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
]

ADJUST_W_BASELINE = ["acceleration_x", "acceleration_y", "acceleration_z"]

POSSIBLE_LABELS = [
    "falling",
    "walking",
    "running",
    "sitting",
    "standing",
    "laying",
    "recover",
]


def add_features(data_frame, rolling_size):
    # Create a copy to avoid modifying the original
    result_df = data_frame.copy()

    # add last two data points per window aswell
    data_types = ["acceleration", "gyroscope"]
    dimensions = ["x", "y", "z"]
    columns = set(result_df.columns)
    for data_type in data_types:
        for dimension in dimensions:
            column_name = f"{data_type}_{dimension}"
            data = result_df[column_name]

            # Calculate rolling window features
            result_df[f"{column_name}_mean"] = data.rolling(
                window=rolling_size, min_periods=1
            ).mean()
            result_df[f"{column_name}_max"] = data.rolling(
                window=rolling_size, min_periods=1
            ).max()
            result_df[f"{column_name}_min"] = data.rolling(
                window=rolling_size, min_periods=1
            ).min()
            result_df[f"{column_name}_std"] = data.rolling(
                window=rolling_size, min_periods=1
            ).std()
            # result_df[f"{column_name}_median"] = data.rolling(
            #     window=rolling_size
            # ).median()

        result_df[f"{data_type}_magnitude"] = np.sqrt(
            result_df[f"{data_type}_x"] ** 2
            + result_df[f"{data_type}_y"] ** 2
            + result_df[f"{data_type}_z"] ** 2
        )

    if "prev_label" not in result_df.columns and "label" in result_df.columns:
        # Get all unique labels

        labels = result_df["label"]

        for l in POSSIBLE_LABELS:
            result_df[f"prev_{l}"] = 0

        counts = {}

        for i in range(len(result_df)):
            # Update window counts
            new_val = labels.iloc[i]
            if pd.notna(new_val):
                counts[new_val] = counts.get(new_val, 0) + 1

            if i >= rolling_size:
                old_val = labels.iloc[i - rolling_size]
                if pd.notna(old_val):
                    counts[old_val] -= 1
            # Set mode if window is complete
            if counts and i >= rolling_size - 1:
                mode_val = max(counts, key=counts.get)
                idx = result_df.index[i]
                result_df.loc[idx, f"prev_{mode_val}"] = 1

    addedColumns = set(result_df.columns) - columns
    return result_df.iloc[rolling_size:], list(addedColumns)


def preprocess_file(path: str, window_size=100):
    df = pd.read_csv(path)
    # df = df[df.label != "none"].dropna()

    # Calculate the baseline as the mean of the first 20 data points for each sensor column
    baseline = df[ADJUST_W_BASELINE].head(20).mean()
    # Subtract the baseline from the sensor columns, for the file, so 0 represents the still state
    # df[SENSOR_COLS] = df[SENSOR_COLS] - baseline
    # print(baseline)

    df = df.ffill()
    df = df.dropna()
    X = df[SENSOR_COLS].astype(float)

    X.loc[:, ADJUST_W_BASELINE] = X[ADJUST_W_BASELINE] - baseline
    X = pd.DataFrame(X, columns=X.columns)

    # feature engineering
    X["label"] = pd.Categorical(df["label"], categories=POSSIBLE_LABELS, ordered=True)
    X, new_features = add_features(X, window_size)
    Y = X["label"]
    X = X.drop("label", axis=1)

    return X, Y, df
