from state_monitor import FallDetectionStateMonitor
from memory import RollingStats, LabelMemory
from common import SENSOR_COLS, POSSIBLE_LABELS
from predict import predict
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


class DataProcessor:
    window: int = 100
    predict_interval: int = 2
    send_interval: int = 10
    baseline_window: int = 10

    state_monitor: FallDetectionStateMonitor
    model: RandomForestClassifier
    scaler: MinMaxScaler

    # Store stats and previous labels for a given window
    rolling_stats: RollingStats
    label_memory: LabelMemory

    # Collect baseline data for starting measurements at 0
    baseline = None
    prev_label_distribution = None
    total_measurements = 0  # Initialize a counter
    feature_cols: list[str] = []  # Features to use for prediction

    def __init__(
        self,
        state_monitor: FallDetectionStateMonitor,
        model: RandomForestClassifier,
        scaler: MinMaxScaler,
        window: int = 100,
        predict_interval: int = 2,
        send_interval: int = 10,
        baseline_window: int = 10,
        feature_cols: list[str] = [],
    ):
        self.window = window
        self.rolling_stats = RollingStats(window)
        self.label_memory = LabelMemory(window, labels=POSSIBLE_LABELS)
        self.state_monitor = state_monitor
        self.model = model
        self.scaler = scaler
        self.predict_interval = predict_interval
        self.send_interval = send_interval
        self.baseline_window = baseline_window
        self.feature_cols = feature_cols

    def process_data(self, json_data: dict):
        data_row = pd.Series(
            json_data,
            index=SENSOR_COLS,
        )

        if self.baseline is None:
            # add data to the rolling stats
            self.rolling_stats.update(data_row)

            if self.rolling_stats.size() >= 10:
                # Compute the baseline averages
                self.baseline = self.rolling_stats.mean()
                self.rolling_stats.clear()
                print("Baseline calculated:")
                print(self.baseline)
                # TODO: Why the sleep?
                time.sleep(2)
            return None

        # If baseline has been calculated, adjust the sensor readings and start sending data
        data_row = data_row - self.baseline
        # scale with trained scaler
        # scaled_data = self.scaler.transform(data_row)

        # add data to the rolling stats
        self.rolling_stats.update(data_row)

        # store the updated values in the json
        json_data = {**data_row.to_dict()}

        # compute magnitudes
        accel_magnitude = np.sqrt(
            data_row["acceleration_x"] ** 2
            + data_row["acceleration_y"] ** 2
            + data_row["acceleration_z"] ** 2
        )
        gyro_magnitude = np.sqrt(
            data_row["gyroscope_x"] ** 2
            + data_row["gyroscope_y"] ** 2
            + data_row["gyroscope_z"] ** 2
        )

        # rename columns of previous distribution to _prev
        prev_labels = (
            {f"prev_{k}": v for k, v in self.prev_label_distribution.items()}
            if self.prev_label_distribution is not None
            else {f"prev_{l}": 0 for l in POSSIBLE_LABELS}
        )

        feature_row = pd.DataFrame(
            {
                "acceleration_magnitude": accel_magnitude,
                "gyroscope_magnitude": gyro_magnitude,
                **data_row[SENSOR_COLS].to_dict(),
                **self.rolling_stats.mean_labeled(),
                **self.rolling_stats.std_labeled(),
                **self.rolling_stats.min_labeled(),
                **self.rolling_stats.max_labeled(),
                **prev_labels,
            },
            index=[0],
        )[self.feature_cols]

        label_dist = None
        # every predict_interval times, make a prediction
        if label_dist is None or self.total_measurements % self.predict_interval == 0:
            label_dist = predict(self.model, feature_row)
            self.label_memory.update(label_dist)
        else:
            label_dist = self.prev_label_distribution

        json_data["current_confidence"] = label_dist.to_dict()
        # Increment the counter
        self.total_measurements += 1

        # Only broadcast the data every send_interval times
        if self.total_measurements % self.send_interval == 0:
            # Calculate the weighted average distribution of the last labels
            avg_label_dist = self.label_memory.averaged_current_label()
            mode_label = self.label_memory.mode()

            json_data["confidence"] = avg_label_dist.to_dict()
            json_data["label"] = mode_label

            self.state_monitor.update_state(mode_label)
            json_data["state"] = self.state_monitor.current_state
            # send a flag to update the logs in history and alert the user
            if self.state_monitor.current_state == [
                "no_movement",
                "fall_notification",
            ]:
                json_data["update_logs"] = True
            return json_data
        # do not respond with json_data
        return None
