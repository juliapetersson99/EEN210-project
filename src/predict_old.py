import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import glob
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib
from scipy.stats import mode

# import tensorflow as tf  # Import TensorFlow


def encode_prev_labels(data_frame):
    # One-hot encode the 'prev_label' column
    encoded_labels = pd.get_dummies(
        data_frame["prev_label"],
        dummy_na=True,
    )
    classes = [
        "falling",
        "walking",
        "running",
        "sitting",
        "standing",
        "laying",
        "recover",
    ]

    for cat in classes:
        if cat not in encoded_labels.columns:
            encoded_labels[cat] = False
        # Drop the original 'prev_label' column and join the encoded labels
    data_frame = data_frame.drop("prev_label", axis=1).join(encoded_labels[classes])
    return data_frame


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
        # output[f"{data_type}_magnitude_std"] = data_frame[f"{data_type}_magnitude"].std()
    if "prev_label" not in data_frame.columns and "label" in data_frame.columns:
        label_cats = pd.Categorical(data_frame["label"])
        data_frame["prev_label"] = (
            pd.Series(label_cats.codes)
            .rolling(window=rolling_size)
            .apply(lambda x: mode(x)[0])
        )
        data_frame["prev_label"] = data_frame["prev_label"].map(
            lambda x: label_cats.categories[int(x)] if pd.notna(x) else None
        )

    addedColumns = set(data_frame.columns) - columns
    return data_frame, list(addedColumns)


def load_data():
    # Get list of all CSV files in the directory
    csv_files = glob.glob("clean_data/*.csv")

    # Read and concatenate all CSV files
    data_frames = [pd.read_csv(file) for file in csv_files]
    data = pd.concat(data_frames, ignore_index=True)

    data = data[data.label != "none"].dropna()
    return data


def train_model(data, window=100):
    # Separate features (sensor data) and labels
    X = data[
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

    # Feature Engineering
    X["label"] = pd.Categorical(data["label"])
    X, new_features = add_features(X, window)
    X = X.iloc[window:].drop("label", axis=1)
    print(X.head(10))

    if "prev_label" in X.columns:
        X = encode_prev_labels(X)

    print(X.head(10))

    # for col in X.columns:
    #     X[col + "_avg"] = X[col].rolling(window=20).mean()
    #     X[col + "_std"] = X[col].rolling(window=20).mean()
    y = data["label"][window:]

    # Split data into training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Initialize and train classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=20)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    return clf, min_max_scaler


def predict(model, scaler, input_df, window=100):
    data = input_df[
        [
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ]
    ]
    arr_scaled = scaler.transform(data)
    X = pd.DataFrame(arr_scaled, columns=data.columns)

    X, new_features = add_features(X, window)

    if "prev_label" in input_df.columns:
        X["prev_label"] = input_df["prev_label"]
        X = encode_prev_labels(X)

    label = model.predict(X[-1:])[0]
    return label
    # df = add_features(input_df)
    # use last row as input for prediction
    # return model.predict(df[-1])


def load_model(name="rf_model_with_scaler"):
    return joblib.load(f"{name}.joblib")


def load_tf_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_tf_model(model, scaler, input_df):
    data = input_df[
        [
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ]
    ]
    arr_scaled = scaler.transform(data)
    X = pd.DataFrame(arr_scaled, columns=data.columns)
    X, new_features = add_features(X, 50)

    X = X[-1:].values.reshape(1, -1, X.shape[1])  # Reshape for LSTM or similar models
    label = model.predict(X)
    return label.argmax(axis=1)[0]  # Assuming the model outputs probabilities


if __name__ == "__main__":
    data = load_data()
    model, scaler = train_model(data)

    joblib.dump((scaler, model), "rf_model_with_scaler.joblib")
    # Example usage of predict function
    # input_data = pd.DataFrame([...])  # Replace with actual input data
    # predictions = predict(model, input_data)
    # print(predictions)

    # Example usage of TensorFlow model
    # tf_model = load_tf_model("path_to_tf_model")
    # tf_predictions = predict_tf_model(tf_model, scaler, input_data)
    # print(tf_predictions)
