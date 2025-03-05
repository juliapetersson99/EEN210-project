import os
import json
from datetime import datetime
import numpy as np
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, Query
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

current_dir = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(current_dir, "UI_bars.html")
with open(html_path, "r") as f:
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

from alert import ConsoleNotifier
from state_monitor import FallDetectionStateMonitor
from storage_classes import FileStorage

# Initialize file storage (the log file will be created or appended to)
storage = FileStorage(file_path="state_changes.log")

notifier = ConsoleNotifier()
state_machine = FallDetectionStateMonitor(notifier, storage)

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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

storage = FileStorage(file_path="state_changes.log")

notifier = ConsoleNotifier()
state_machine = FallDetectionStateMonitor(notifier, storage)

@app.get("/logs")
def get_logs(patientId: str = Query(...)):
    # Retrieve logs for the given patient
    events = storage.read_events_for_patient(patientId)
    return events

app.mount("/static", StaticFiles(directory=current_dir), name="static")
@app.get("/main.css")
async def get_css():
    return FileResponse(os.path.join(current_dir, "main.css"))

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
    patient_id = websocket.query_params.get("patientId", "unknown")
                                            
    df = None
    state_machine.set_patinet_id(patient_id)
    await websocket_manager.connect(websocket)
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



    # Baseline calibration variables and buffers
    baselineCalculated = False
    accelBaseline = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    gyroBaseline = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    accelBuffer = {'x': [], 'y': [], 'z': []}
    gyroBuffer = {'x': [], 'y': [], 'z': []}

    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()  # Start timing

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction
            raw_data = list(json_data.values())

            ax = float(json_data["acceleration_x"])
            ay = float(json_data["acceleration_y"])
            az = float(json_data["acceleration_z"])
            gx = float(json_data["gyroscope_x"])
            gy = float(json_data["gyroscope_y"])
            gz = float(json_data["gyroscope_z"])

            


            if not baselineCalculated:
                accelBuffer['x'].append(ax)
                accelBuffer['y'].append(ay)
                accelBuffer['z'].append(az)


                gyroBuffer['x'].append(gx)
                gyroBuffer['y'].append(gy)
                gyroBuffer['z'].append(gz)
                
                if len(accelBuffer['x']) >= 10:
                    # Compute the baseline averages
                    accelBaseline['x'] = np.mean(accelBuffer['x'])
                    accelBaseline['y'] = np.mean(accelBuffer['y'])
                    accelBaseline['z'] = np.mean(accelBuffer['z'])


                    gyroBaseline['x'] = np.mean(gyroBuffer['x'])
                    gyroBaseline['y'] = np.mean(gyroBuffer['y'])
                    gyroBaseline['z'] = np.mean(gyroBuffer['z'])
                    baselineCalculated = True
                    print("Baseline calculated:", accelBaseline, gyroBaseline)
                    time.sleep(2)
            
            # If baseline has been calculated, adjust the sensor readings and start sending data
            else:
                ax = (ax - accelBaseline['x'])
                ay =(ay - accelBaseline['y'])
                az = (az - accelBaseline['z'])

                gx = (gx - gyroBaseline['x'])
                gy = (gy - gyroBaseline['y'])
                gz = (gz - gyroBaseline['z'])

                #add to adjusted data
                json_data["acceleration_x"] = ax
                json_data["acceleration_y"] = ay
                json_data["acceleration_z"] = az
                json_data["gyroscope_x"] = gx
                json_data["gyroscope_y"] = gy
                json_data["gyroscope_z"] = gz
                #print("Adjusted values:", ax, ay, az, gx, gy, gz)

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

                feature_df = pd.DataFrame([feature_row])
                feature_df["timestamp"] = pd.Timestamp.now()
                feature_df["label"] = None  # Set to None if you don't have a label yet
                feature_df["prev_label"] = None  # Set to None if you don't have a label yet
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
                    #print(df)
                    label = df["label"].iloc[-2]  # take last label
                json_data["label"] = current_label or label
                df.loc[df.index[-1], "label"] = current_label or label

                # Increment the counter
                total_measurements += 1

                # Only broadcast the data every send_interval times
                if total_measurements % send_interval == 0:
                    # Calculate the distribution of the last 50 labels
                    label_distribution = df["label"].value_counts(normalize=True).to_dict()


                    json_data["confidence"] = label_distribution

                    end_time = time.time()  # End timing
                    json_data["processing_time"] = (
                        end_time - start_time
                    )  # Add processing time to the data
                    print(f"Processing time: {json_data['processing_time']:.3f} seconds")
                    state_machine.update_state(label)
                    json_data["state"] = state_machine.current_state
                    print(json_data)
                    # send a flag to update the logs in history and alert the user
                    if state_machine.current_state == ["no_movement", "fall_notification"]:
                        json_data["update_logs"] = True
                    await websocket_manager.broadcast_message(json.dumps(json_data))
                end_time = time.time()  # End timing
                print(f"Processing time: {end_time-start_time:.3f} seconds")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
