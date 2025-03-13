import pandas as pd
from process import DataProcessor
from alert import ConsoleNotifier
from storage_classes import FileStorage
from state_monitor import FallDetectionStateMonitor

from predict import load_model
from common import preprocess_file
from fhir import demo_patient

# Prepare dummy CSV data path (adjust the file name as needed)
csv_file = "../clean_data/Julia_maskin_clean.csv"  # This file should be located at /c:/dev/python/EEN210-project/tests/test_data.csv


def test_process():
    """
    Test the process function compared to the training preprocessing.
    """
    # Read CSV file into a DataFrame and preprocess with the training code
    X, y, df = preprocess_file(csv_file, window_size=20)

    test_datapoints = 7000

    print(X.head(test_datapoints))

    # Instantiate the state monitor with its dependencies
    notifier = ConsoleNotifier()
    storage = FileStorage(file_path="test_state_changes.log")
    state_monitor = FallDetectionStateMonitor(notifier, storage)
    state_monitor.set_patient(demo_patient())

    # Load the model, scaler, and feature columns with joblib as in server
    model, scaler, feature_cols = load_model("../final_model")

    dp = DataProcessor(
        state_monitor=state_monitor,
        model=model,
        scaler=scaler,
        window=20,
        predict_interval=1,
        send_interval=1,
        baseline_window=20,
        feature_cols=feature_cols,
    )

    outputs = []  # Initialize list for outputs
    # Process data row by row
    for _, row in df.iloc[:test_datapoints].iterrows():
        json_data = row.to_dict()
        output = dp.process_data(json_data)
        if output is not None:
            outputs.append(output)

    # Convert outputs list into a DataFrame and display it
    if outputs:
        results_df = pd.DataFrame(outputs)
        print(results_df)

        # Make sure both axes have all possible labels
        confusion_matrix = pd.crosstab(
            y[:test_datapoints],
            results_df["label"],
            rownames=["Actual"],
            colnames=["Predicted"],
            dropna=False,
        )
        print(confusion_matrix)
    else:
        print("No outputs generated.")

    return X, results_df


if __name__ == "__main__":
    X, df = test_process()
