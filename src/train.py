import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
import glob
from sklearn.ensemble import RandomForestClassifier


def add_features(data_frame, rolling_size):

    # add last two data points per window aswell
    data_types = ["acceleration", "gyroscope"]
    dimensions = ["x", "y", "z"]
    columns = set(data_frame.columns)
    for data_type in data_types:
        for dimension in dimensions:
            column_name = f"{data_type}_{dimension}"
            data = data_frame[column_name]

            data_frame[f"{column_name}_mean"] = data.rolling(window=rolling_size).mean()
            data_frame[f"{column_name}_max"] = data.rolling(window=rolling_size).max()
            data_frame[f"{column_name}_min"] = data.rolling(window=rolling_size).min()
            data_frame[f"{column_name}_std"] = data.rolling(window=rolling_size).std()
            data_frame[f"{column_name}_median"] = data.rolling(
                window=rolling_size
            ).median()

        data_frame[f"{data_type}_magnitude"] = np.sqrt(
            data_frame[f"{data_type}_x"] ** 2
            + data_frame[f"{data_type}_y"] ** 2
            + data_frame[f"{data_type}_z"] ** 2
        )

    if "prev_label" not in data_frame.columns and "label" in data_frame.columns:
        # Get all unique labels
        unique_labels = [
            "falling",
            "walking",
            "running",
            "sitting",
            "standing",
            "laying",
            "recover",
        ]
        labels = data_frame["label"]

        for l in unique_labels:
            data_frame[f"prev_{l}"] = 0

        counts = {}

        for i in range(len(data_frame)):
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
                data_frame.loc[i, f"prev_{mode_val}"] = 1

    addedColumns = set(data_frame.columns) - columns
    return data_frame.iloc[rolling_size:], list(addedColumns)


def preprocess_file(path: str, window_size=100) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.label != "none"].dropna()

    # Separate features (sensor data) and labels
    X = df[
        [
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ]
    ]

    min_max_scaler = MinMaxScaler()
    arr_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(arr_scaled, columns=X.columns)

    # feature engineering
    X["label"] = pd.Categorical(df["label"])
    X, new_features = add_features(X, window_size)
    Y = X["label"]
    X = X.drop("label", axis=1)

    return X, Y


def train_model(X, y, window=100, settings=dict(n_estimators=100, max_depth=20)):
    clf = RandomForestClassifier(**settings)
    clf.fit(X, y)
    return clf


def validate_model(clf, X_test, y_test):

    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")

    # confusion matrix
    all_labels = sorted(clf.classes_)  # All possible labels from the classifier

    # Make sure both axes have all possible labels
    confusion_matrix = pd.crosstab(
        y_test, y_pred, rownames=["Actual"], colnames=["Predicted"], dropna=False
    )

    # Ensure all labels are present in both rows and columns
    for label in all_labels:
        if label not in confusion_matrix.index:
            confusion_matrix.loc[label] = 0
        if label not in confusion_matrix.columns:
            confusion_matrix[label] = 0

    # Sort to ensure consistent order
    confusion_matrix = confusion_matrix.loc[all_labels, all_labels]
    print("Confusion matrix")
    print(confusion_matrix)

    # feature importance
    # feature_importance = clf.feature_importances_
    # print("Feature importance")
    # print("\n".join(f"{f}: {i}" for f, i in zip(X_test.columns, feature_importance)))

    # mean square error
    y_pred = clf.predict_proba(X_test)
    y_true = pd.get_dummies(y_test)
    # Ensure all classes are in the one-hot encoding
    for cls in all_labels:
        if cls not in y_true.columns:
            y_true[cls] = 0

    # Reorder columns to match classifier's classes_ order
    y_true = y_true[all_labels]
    mse = np.mean((y_true - y_pred) ** 2)
    print(f"Mean square error: {mse}")

    return accuracy, confusion_matrix, mse


def cross_validation_testing(
    model_settings=dict(n_estimators=100, max_depth=20), folder="data", num_folds=5
):
    csv_files = glob.glob(f"{folder}/*.csv")

    # Read and preprocess all CSV files
    data = list(map(preprocess_file, csv_files))
    X, y = zip(*data)

    mean_accuracy = 0
    mean_mse = 0
    mean_confusion_matrix = np.zeros((7, 7))

    # 5-fold cross validation
    fold_size = len(data) // num_folds
    print(fold_size)
    for i in range(num_folds):
        start = i * fold_size
        end = (i + 1) * fold_size

        X_test = pd.concat(X[start:end])
        y_test = pd.concat(y[start:end])

        X_train = pd.concat(X[:start] + X[end:])
        y_train = pd.concat(y[:start] + y[end:])

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

for n in [10, 50, 100, 200]:
    print(f"n_estimators = {n}")
    cross_validation_testing(model_settings=dict(n_estimators=n, max_depth=20))
