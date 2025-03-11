import joblib
import pandas as pd
from common import POSSIBLE_LABELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


def load_model(name="final_model") -> (RandomForestClassifier, MinMaxScaler):
    return joblib.load(f"{name}.joblib")


def encode_prev_labels(data_frame):
    # One-hot encode the 'prev_label' column
    encoded_labels = pd.get_dummies(
        data_frame["prev_label"],
        dummy_na=True,
    )

    for cat in POSSIBLE_LABELS:
        if cat not in encoded_labels.columns:
            encoded_labels[cat] = False
        # Drop the original 'prev_label' column and join the encoded labels
    data_frame = data_frame.drop("prev_label", axis=1).join(
        encoded_labels[POSSIBLE_LABELS]
    )
    return data_frame


def predict_proba(model, input_data):
    input_df = (
        input_data.to_frame().T if isinstance(input_data, pd.Series) else input_data
    )

    if "prev_label" in input_df.columns:
        input_df = encode_prev_labels(input_df)

    dist = model.predict_proba(input_df)
    labeled_dist = dict(zip(model.classes_, dist[0]))

    return pd.Series(labeled_dist)


def predict(model, input_data):
    input_df = (
        input_data.to_frame().T if isinstance(input_data, pd.Series) else input_data
    )

    if "prev_label" in input_df.columns:
        input_df = encode_prev_labels(input_df)

    labeled_dist = {k: 0 for k in POSSIBLE_LABELS}
    label = model.predict(input_df)[0]
    labeled_dist[label] = 1

    return pd.Series(labeled_dist)
