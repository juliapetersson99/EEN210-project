import os
import json
from datetime import datetime
import numpy as np
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
from predict import add_features, load_data, train_model, predict
import time

app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(
    "./src/UI_bars.html", "r"
) as f:  # Path is C:\Users\julia\OneDrive\Avslutade kurser\Skrivbord\VSCode-file\Fall_Detection_project
    html = f.read()


class DataProcessor:
    def __init__(self):
        self.data_buffer = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"fall_data_{timestamp}.csv"

    def add_data(self, data):
        self.data_buffer.append(data)

    def save_to_csv(self):
        df = pd.DataFrame.from_dict(self.data_buffer)
        self.data_buffer = []
        # Append the new row to the existing DataFrame
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        # print(f"DataFrame saved to {self.file_path}")


data_processor = DataProcessor()

window = 100
send_interval = 15  # Define the interval for sending data
predict_interval = 2


def load_model():
    # you should modify this function to return your model
    print("Loading model...")
    # data = load_data()
    # print("Training model...")
    # model, scaler = train_model(data)
    scaler, model = joblib.load("rf_model_with_scaler.joblib")
    print("Model loaded successfully")
    return model, scaler


def predict_label(model=None, scaler=None, df=None):
    global window
    # you should modify this to return the label
    if model is not None:
        return predict(model, scaler, df, window=window)
    return 0


from collections import deque
import math


class RollingStats:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.data_deque = deque()
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, new_value: float):
        """
        Add a new data point into the rolling window.
        Remove the oldest data point if we're over capacity.
        """
        # Add new sample
        self.data_deque.append(new_value)
        self.sum += new_value
        self.sum_sq += new_value * new_value

        # Pop oldest if over window size
        if len(self.data_deque) > self.window_size:
            old_value = self.data_deque.popleft()
            self.sum -= old_value
            self.sum_sq -= old_value * old_value

    def mean(self) -> float:
        """
        Returns the rolling mean of the current window.
        """
        current_size = len(self.data_deque)
        if current_size == 0:
            return 0.0
        return self.sum / current_size

    def variance(self) -> float:
        """
        Returns the rolling sample variance of the current window.
        """
        current_size = len(self.data_deque)
        if current_size < 2:
            return 0.0
        # sample variance = (sum of x^2 - (sum of x)^2 / n ) / (n-1)
        return (self.sum_sq - (self.sum * self.sum) / current_size) / (current_size - 1)

    def std(self) -> float:
        return math.sqrt(self.variance() + 0.0001) if len(self.data_deque) > 1 else 0.0


class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("WebSocket disconnected")

    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                # Handle disconnect if needed
                self.disconnect(connection)


websocket_manager = WebSocketManager()
model, scaler = load_model()


@app.get("/")
async def get():
    return HTMLResponse(html)


current_label = None


@app.get("/collect/{label}")
async def collect_data(label: str):
    global current_label
    print("Label changed to:", label)
    current_label = label

    return {"message": "Data collected"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_label, window, send_interval, predict_interval
    await websocket_manager.connect(websocket)
    df = None
    total_measurements = 0  # Initialize a counter

    # Acceleration rolling stats (x, y, z, magnitude)
    rolling_stats_ax = RollingStats(window)
    rolling_stats_ay = RollingStats(window)
    rolling_stats_az = RollingStats(window)
    rolling_stats_am = RollingStats(window)  # acceleration magnitude

    # Gyroscope rolling stats (x, y, z, magnitude)
    rolling_stats_gx = RollingStats(window)
    rolling_stats_gy = RollingStats(window)
    rolling_stats_gz = RollingStats(window)
    rolling_stats_gm = RollingStats(window)  # gyroscope magnitude

    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()  # Start timing

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction
            raw_data = list(json_data.values())

            # Add time stamp to the last received data
            # json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            # data_processor.add_data(json_data)
            # this line save the recent 100 samples to the CSV file. you can change 100 if you want.
            # if len(data_processor.data_buffer) >= 100:
            #    data_processor.save_to_csv()

            # Extract raw sensor values
            ax = float(json_data["acceleration_x"])
            ay = float(json_data["acceleration_y"])
            az = float(json_data["acceleration_z"])
            gx = float(json_data["gyroscope_x"])
            gy = float(json_data["gyroscope_y"])
            gz = float(json_data["gyroscope_z"])

            accel_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
            gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)

            # Update rolling stats (so we de not need to keep computing rolling average)
            rolling_stats_ax.update(ax)
            rolling_stats_ay.update(ay)
            rolling_stats_az.update(az)
            rolling_stats_am.update(accel_magnitude)

            rolling_stats_gx.update(gx)
            rolling_stats_gy.update(gy)
            rolling_stats_gz.update(gz)
            rolling_stats_gm.update(gyro_magnitude)

            # Update each rolling stats object
            ax_mean = rolling_stats_ax.mean()
            ax_std = rolling_stats_ax.std()
            ay_mean = rolling_stats_ay.mean()
            ay_std = rolling_stats_ay.std()
            az_mean = rolling_stats_az.mean()
            az_std = rolling_stats_az.std()
            am_mean = rolling_stats_am.mean()
            am_std = rolling_stats_am.std()

            gx_mean = rolling_stats_gx.mean()
            gx_std = rolling_stats_gx.std()
            gy_mean = rolling_stats_gy.mean()
            gy_std = rolling_stats_gy.std()
            gz_mean = rolling_stats_gz.mean()
            gz_std = rolling_stats_gz.std()
            gm_mean = rolling_stats_gm.mean()
            gm_std = rolling_stats_gm.std()

            # Now compute rolling means/stds
            ax_mean = rolling_stats_ax.mean()
            ax_std = rolling_stats_ax.std()

            ay_mean = rolling_stats_ay.mean()
            ay_std = rolling_stats_ay.std()

            az_mean = rolling_stats_az.mean()
            az_std = rolling_stats_az.std()

            gx_mean = rolling_stats_gx.mean()
            gx_std = rolling_stats_gx.std()

            gy_mean = rolling_stats_gy.mean()
            gy_std = rolling_stats_gy.std()

            gz_mean = rolling_stats_gz.mean()
            gz_std = rolling_stats_gz.std()
            feature_row = {
                "acceleration_x": ax,
                "acceleration_y": ay,
                "acceleration_z": az,
                "acceleration_magnitude": accel_magnitude,
                "acceleration_x_mean": ax_mean,
                "acceleration_x_std": ax_std,
                "acceleration_y_mean": ay_mean,
                "acceleration_y_std": ay_std,
                "acceleration_z_mean": az_mean,
                "acceleration_z_std": az_std,
                # "acceleration_mag_mean": am_mean,
                # "acceleration_mag_std": am_std,
                "gyroscope_x": gx,
                "gyroscope_y": gy,
                "gyroscope_z": gz,
                "gyroscope_magnitude": gyro_magnitude,
                "gyroscope_x_mean": gx_mean,
                "gyroscope_x_std": gx_std,
                "gyroscope_y_mean": gy_mean,
                "gyroscope_y_std": gy_std,
                "gyroscope_z_mean": gz_mean,
                "gyroscope_z_std": gz_std,
                # "gyroscope_mag_mean": gm_mean,
                # "gyroscope_mag_std": gm_std,
            }

            # Add a timestamp

            # newDataDf = pd.DataFrame(json_data, index=[0])
            # newDataDf["timestamp"] = pd.Timestamp.now()
            # newDataDf["label"] = None
            # newDataDf["prev_label"] = None

            # Update rolling stats

            #
            # feature_df = pd.concat([feature_df, pd.DataFrame([feature_row])], ignore_index=True)
            feature_df = pd.DataFrame([feature_row])
            feature_df["timestamp"] = pd.Timestamp.now()
            feature_df["label"] = None  # Set to None if you don't have a label yet
            feature_df["prev_label"] = None  # Set to None if you don't have a label yet

            # Example: let's define "label" either from a user or from a model prediction
            # If you're collecting data for labeled training, you might have current_label set globally.
            label = current_label or "unknown"

            if df is None:
                df = feature_df
            else:
                # find the most common in the previous window
                if total_measurements % 50 == 0:
                    feature_df["prev_label"] = df["label"].mode().values[0]
                df = pd.concat([df, feature_df], ignore_index=True)
                if len(df) > window:
                    df = df[1:]

            """  
            In this line we use the model to predict the labels.
            Right now it only return 0.
            You need to modify the predict_label function to return the true label
            """
            if total_measurements % predict_interval == 0:
                label = predict_label(model, scaler, df)
            else:
                print(df)
                label = df["label"].iloc[-2]  # take last label
            json_data["label"] = current_label or label
            df.loc[df.index[-1], "label"] = current_label or label

            # Increment the counter
            total_measurements += 1

            # Only broadcast the data every send_interval times
            if total_measurements % send_interval == 0:
                # Calculate the distribution of the last 50 labels
                label_distribution = df["label"].value_counts(normalize=True).to_dict()

                # print the last data in the terminal
                print(json_data)
                print(
                    label_distribution  # .to_string(header=["Label", "Proportion"], index=True)
                )
                json_data["confidence"] = label_distribution

                end_time = time.time()  # End timing
                json_data["processing_time"] = (
                    end_time - start_time
                )  # Add processing time to the data
                print(f"Processing time: {json_data['processing_time']:.3f} seconds")

                # broadcast the last data to webpage
                await websocket_manager.broadcast_message(json.dumps(json_data))
            end_time = time.time()  # End timing
            print(f"Processing time: {end_time-start_time:.3f} seconds")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
