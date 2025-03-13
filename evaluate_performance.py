import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score

def add_features(data_frame, rolling_size):
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
            data_frame[f"{column_name}_median"] = data.rolling(window=rolling_size).median()

        data_frame[f"{data_type}_magnitude"] = np.sqrt(
            data_frame[f"{data_type}_x"] ** 2
            + data_frame[f"{data_type}_y"] ** 2
            + data_frame[f"{data_type}_z"] ** 2
        )

    if "prev_label" not in data_frame.columns and "label" in data_frame.columns:
        unique_labels = ["falling", "walking", "running", "sitting", "standing", "laying", "recover"]
        labels = data_frame["label"]

        for l in unique_labels:
            data_frame[f"prev_{l}"] = 0

        counts = {}

        for i in range(len(data_frame)):
            new_val = labels.iloc[i]
            if pd.notna(new_val):
                counts[new_val] = counts.get(new_val, 0) + 1

            if i >= rolling_size:
                old_val = labels.iloc[i - rolling_size]
                if pd.notna(old_val):
                    counts[old_val] -= 1
            
            if counts and i >= rolling_size - 1:
                mode_val = max(counts, key=counts.get)
                data_frame.loc[i, f"prev_{mode_val}"] = 1

    addedColumns = set(data_frame.columns) - columns
    return data_frame.iloc[rolling_size:], list(addedColumns)

def preprocess_file(path: str, window_size=100) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.label != "none"].dropna()

    X = df[
        ["acceleration_x", "acceleration_y", "acceleration_z",
         "gyroscope_x", "gyroscope_y", "gyroscope_z"]
    ]

    min_max_scaler = MinMaxScaler()
    arr_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(arr_scaled, columns=X.columns)

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

    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    
    # Balanced Accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Macro F1-Score
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Confusion Matrix
    all_labels = sorted(clf.classes_)
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"], dropna=False)

    for label in all_labels:
        if label not in confusion_matrix.index:
            confusion_matrix.loc[label] = 0
        if label not in confusion_matrix.columns:
            confusion_matrix[label] = 0

    confusion_matrix = confusion_matrix.loc[all_labels, all_labels]

    # Mean Squared Error
    y_pred_prob = clf.predict_proba(X_test)
    y_true = pd.get_dummies(y_test)

    for cls in all_labels:
        if cls not in y_true.columns:
            y_true[cls] = 0

    y_true = y_true[all_labels]
    mse = np.mean((y_true - y_pred_prob) ** 2)

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix)
    print(f"Mean square error: {mse:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    return accuracy, bal_acc, macro_f1, confusion_matrix, mse, mcc

def cross_validation_testing(model_settings=dict(n_estimators=100, max_depth=20), folder="data", num_folds=5):
    csv_files = glob.glob(f"{folder}/*.csv")
    data = list(map(preprocess_file, csv_files))
    X, y = zip(*data)

    mean_accuracy = 0
    mean_bal_acc = 0
    mean_macro_f1 = 0
    mean_mse = 0
    mean_mcc = 0
    mean_confusion_matrix = np.zeros((7, 7))

    fold_size = len(data) // num_folds
    for i in range(num_folds):
        start, end = i * fold_size, (i + 1) * fold_size
        X_test, y_test = pd.concat(X[start:end]), pd.concat(y[start:end])
        X_train, y_train = pd.concat(X[:start] + X[end:]), pd.concat(y[:start] + y[end:])

        print(f"Training Fold {i}")
        clf = train_model(X_train, y_train, settings=model_settings)

        print(f"Validating Fold {i}")
        acc, bal_acc, macro_f1, conf_matrix, mse, mcc = validate_model(clf, X_test, y_test)

        mean_accuracy += acc
        mean_bal_acc += bal_acc
        mean_macro_f1 += macro_f1
        mean_mse += mse
        mean_mcc += mcc
        mean_confusion_matrix += conf_matrix

    print(f"Mean Accuracy: {mean_accuracy / num_folds:.4f}")
    print(f"Mean Balanced Accuracy: {mean_bal_acc / num_folds:.4f}")
    print(f"Mean Macro F1-Score: {mean_macro_f1 / num_folds:.4f}")
    print(f"Mean MSE: {mean_mse / num_folds:.4f}")
    print(f"Mean MCC: {mean_mcc / num_folds:.4f}")
    print("Mean Confusion Matrix:")
    print(mean_confusion_matrix / num_folds)
    print("-------------------------------------------------------------")

cross_validation_testing()
