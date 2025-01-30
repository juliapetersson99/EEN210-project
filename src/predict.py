import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import glob


def load_data():
    # Get list of all CSV files in the directory
    csv_files = glob.glob("*.csv")

    # Read and concatenate all CSV files
    data_frames = [pd.read_csv(file) for file in csv_files]
    data = pd.concat(data_frames, ignore_index=True)

    data = data[data.label != "none"].dropna()
    return data


def train_model(data):
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
    y = data["label"]

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

    return clf


def predict(model, input_data):
    return model.predict(input_data)


if __name__ == "__main__":
    data = load_data()
    model = train_model(data)
    # Example usage of predict function
    # input_data = pd.DataFrame([...])  # Replace with actual input data
    # predictions = predict(model, input_data)
    # print(predictions)
