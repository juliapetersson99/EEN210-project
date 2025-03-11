import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
from common import SENSOR_COLS, POSSIBLE_LABELS


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
    df = df.ffill()
    df = df.dropna()
    # df = df[df.label != "none"].dropna()

    # Calculate the baseline as the mean of the first 20 data points for each sensor column
    # baseline = df[SENSOR_COLS].head(50).mean()
    # Subtract the baseline from the sensor columns, for the file, so 0 represents the still state
    # df[SENSOR_COLS] = df[SENSOR_COLS] - baseline

    # Separate features (sensor data) and labels
    X = df[SENSOR_COLS]
    # min_max_scaler = MinMaxScaler()
    # arr_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=X.columns)

    # feature engineering
    X["label"] = pd.Categorical(df["label"])
    X, new_features = add_features(X, window_size)
    Y = X["label"]
    X = X.drop("label", axis=1)

    return X, Y, df


def train_model(X, y, window=100, settings=dict(n_estimators=100, max_depth=20)):
    clf = RandomForestClassifier(**settings)
    # print(X[X.isna().any(axis=1)])
    # print(y)
    clf.fit(X, y)
    return clf


def validate_model(clf, X_test, y_test):

    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")

    # confusion matrix

    # Make sure both axes have all possible labels
    confusion_matrix = pd.crosstab(
        y_test, y_pred, rownames=["Actual"], colnames=["Predicted"], dropna=False
    )

    # Ensure all labels are present in both rows and columns
    for label in POSSIBLE_LABELS:
        if label not in confusion_matrix.index:
            confusion_matrix.loc[label] = 0
        if label not in confusion_matrix.columns:
            confusion_matrix[label] = 0

    # Sort to ensure consistent order
    confusion_matrix = confusion_matrix.loc[POSSIBLE_LABELS, POSSIBLE_LABELS]
    print("Confusion matrix")
    print(confusion_matrix)
    # normalize the confusion matrix
    norm_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0)
    print("Normalized confusion matrix")
    print(norm_confusion_matrix)

    print("Misclassified samples")
    for l in POSSIBLE_LABELS:
        print(
            f"{l}: {confusion_matrix.loc[l, :].sum() - confusion_matrix.loc[l, l]} | {(1 - norm_confusion_matrix.loc[l, l]):.4%}"
        )

    # feature importance
    # feature_importance = clf.feature_importances_
    # print("Feature importance")
    # print("\n".join(f"{f}: {i}" for f, i in zip(X_test.columns, feature_importance)))

    # mean square error
    y_pred = clf.predict_proba(X_test)
    y_true = pd.get_dummies(y_test)
    # Ensure all classes are in the one-hot encoding
    for cls in POSSIBLE_LABELS:
        if cls not in y_true.columns:
            y_true[cls] = 0

    # Reorder columns to match classifier's classes_ order
    y_true = y_true[POSSIBLE_LABELS]
    mse = np.mean((y_true - y_pred) ** 2)
    print(f"Mean square error: {mse}")

    return accuracy, norm_confusion_matrix, mse


def cross_validation_testing(
    model_settings=dict(n_estimators=100, max_depth=20), folder="data", num_folds=5
):
    csv_files = glob.glob(f"{folder}/*.csv")
    # randomize the order of the files
    random.shuffle(csv_files)

    # Read and preprocess all CSV files
    data = list(map(preprocess_file, csv_files))
    X, y, _ = zip(*data)


    mean_accuracy = 0
    mean_mse = 0
    mean_confusion_matrix = np.zeros((7, 7))

    # 5-fold cross validation
    fold_size = len(data) // num_folds
    print(fold_size)
    for i in range(num_folds):
        min_max_scaler = MinMaxScaler()
        start = i * fold_size
        end = (i + 1) * fold_size

        X_test = pd.concat(X[start:end])
        y_test = pd.concat(y[start:end])

        X_train = pd.concat(X[:start] + X[end:])
        y_train = pd.concat(y[:start] + y[end:])

        X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(min_max_scaler.transform(X_test), columns=X_test.columns)

        # shuffle the data (works because we have no temporal dependencies in the model itself)
        idx = np.random.permutation(len(X_train))
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]

        print(f"Training Fold {i}")
        print(f"num samples: {len(X_train)}, test samples: {len(X_test)}")
        clf = train_model(X_train, y_train, settings=model_settings)

        print(f"Validating Fold {i}")
        acc, confusion_matrix, mse = validate_model(clf, X_test, y_test)
        mean_accuracy += acc
        mean_mse += mse
        mean_confusion_matrix += confusion_matrix

    print(f"Mean accuracy: {mean_accuracy / num_folds}")
    print(f"Mean MSE: {mean_mse / num_folds}")
    print("Mean confusion matrix")
    print(mean_confusion_matrix / num_folds)




# cross_validation_testing()

# for n in [50, 100, 200]:
#     print(f"n_estimators = {n}")
#     cross_validation_testing(model_settings=dict(n_estimators=n, max_depth=20))

# cross_validation_testing(folder="clean_data")

# %% Train model


def train_final_model(
    folder="data", model_settings=dict(n_estimators=100, max_depth=20)
):
    print("Loading and preprocessing data")
    # Load the data
    csv_files = glob.glob(f"{folder}/*.csv")

    # Read and preprocess all CSV files
    data = list(map(preprocess_file, csv_files))
    # join all the data
    X, y, dfs = zip(*data)

    print("Training Scaler")
    # train a scaler with the raw data
    # combined_df = pd.concat(dfs, ignore_index=True)[SENSOR_COLS]
    print("Training model")
    # join data
    X = pd.concat(X)
    y = pd.concat(y)

    min_max_scaler = MinMaxScaler()
    X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

    # shuffle the data
    idx = np.random.permutation(len(X))
    X = X.iloc[idx]
    y = y.iloc[idx]

    clf = train_model(X, y, settings=model_settings)
    print("Model trained")

    # save the model
    joblib.dump((clf, min_max_scaler, X.columns), "final_model.joblib")
    print("Model saved")

    # validate the model
    print("Training stats:")
    validate_model(clf, X, y)

train_final_model(folder="clean_data")
#for n_estimators in [50, 100, 200]:
 #   for max_depth in [10, 20, 30]:
 #       print(f"Testing with n_estimators={n_estimators} and max_depth={max_depth}")
 #       model_settings = dict(n_estimators=n_estimators, max_depth=max_depth)
 #       cross_validation_testing(model_settings=model_settings, folder="clean_data")