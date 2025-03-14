import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
from common import POSSIBLE_LABELS, preprocess_file


def train_model(X, y, settings=dict(n_estimators=100, max_depth=20)):
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
    norm_confusion_matrix = confusion_matrix.div(
        confusion_matrix.sum(axis=0), axis=1
    ).fillna(0)
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

    return accuracy, confusion_matrix, mse


def cross_validation_testing(
    model_settings=dict(n_estimators=100, max_depth=20),
    folder="data",
    num_folds=5,
    window_size=100,
):
    csv_files = glob.glob(f"{folder}/*.csv")
    # randomize the order of the files
    random.shuffle(csv_files)

    # Read and preprocess all CSV files
    data = list(map(lambda f: preprocess_file(f, window_size=window_size), csv_files))
    X, y, _ = zip(*data)

    mean_accuracy = 0
    mean_mse = 0
    total_confusion_matrix = np.zeros((7, 7))

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

        X_train = pd.DataFrame(
            min_max_scaler.fit_transform(X_train), columns=X_train.columns
        )
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
        total_confusion_matrix += confusion_matrix

    mean_accuracy /= num_folds
    mean_mse /= num_folds
    mean_confusion_matrix = total_confusion_matrix / total_confusion_matrix.sum(axis=0)

    print(f"Mean accuracy: {mean_accuracy}")
    print(f"Mean MSE: {mean_mse}")
    print("Total confusion matrix")
    print(total_confusion_matrix)
    print("Mean confusion matrix")
    print(mean_confusion_matrix)

    # create table of precision, recall and f1 per class from confusion matrix
    stat_data = []
    for label in POSSIBLE_LABELS:
        precision = (
            total_confusion_matrix.loc[label, label]
            / total_confusion_matrix.loc[label, :].sum()
        )
        recall = (
            total_confusion_matrix.loc[label, label]
            / total_confusion_matrix.loc[:, label].sum()
        )

        stat_table = stat_data.append(
            {
                "Label": label,
                "Precision": precision,
                "Recall": recall,
                "F1": 2 * precision * recall / (precision + recall),
            },
        )
    stat_table = pd.DataFrame(stat_data)
    stat_table.set_index("Label", inplace=True)

    print(stat_table)
    print("Mean stats")
    print(stat_table.mean())
    # latex table
    print(stat_table.to_latex())

    return mean_accuracy, total_confusion_matrix, mean_mse


# metrics_data = []

# for n_estimators in [50, 100, 200]:
#     for max_depth in [10, 15, 20]:
#         for window_size in [20, 50, 100, 200]:
#             print(
#                 f"Testing with n_estimators={n_estimators}, window={window_size} and max_depth={max_depth}"
#             )
#             model_settings = dict(n_estimators=n_estimators, max_depth=max_depth)
#             acc, confusion_matrix, mean_mse = cross_validation_testing(
#                 model_settings=model_settings,
#                 folder="clean_data",
#                 window_size=window_size,
#             )
#             metrics_data.append(
#                 {
#                     "n_estimators": n_estimators,
#                     "max_depth": max_depth,
#                     "window": window_size,
#                     "accuracy": acc,
#                     "mse": mean_mse,
#                     **{
#                         f"{l}_correct": confusion_matrix.loc[l, l]
#                         for l in POSSIBLE_LABELS
#                     },
#                 }
#             )

# metrics_df = pd.DataFrame(metrics_data)
# metrics_df.to_csv("cv_metrics.csv", index=False)


# %% Train model


def train_final_model(
    folder="data", model_settings=dict(n_estimators=100, max_depth=20), window_size=100
):
    print("Loading and preprocessing data")
    # Load the data
    csv_files = glob.glob(f"{folder}/*.csv")

    # Read and preprocess all CSV files
    data = list(map(lambda f: preprocess_file(f, window_size=window_size), csv_files))
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

    # print feature importance
    print("Feature importance")
    feature_importance = clf.feature_importances_
    print("\n".join(f"{f}: {i}" for f, i in zip(X.columns, feature_importance)))


train_final_model(
    folder="clean_data",
    model_settings=dict(n_estimators=50, max_depth=10),
    window_size=20,
)

# Generate statistics for final parameters.

acc, confusion_matrix, mse = cross_validation_testing(
    folder="clean_data",
    model_settings=dict(n_estimators=50, max_depth=10),
    window_size=20,
)

# plot confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt=".0f", cmap="Blues")
plt.savefig("confusion_matrix.png")
plt.show()
