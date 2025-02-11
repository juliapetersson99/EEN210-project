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
    "./src/index.html", "r"
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
    # you should modify this to return the label
    if model is not None:
        return predict(model, scaler, df)
    return 0


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
    global current_label
    await websocket_manager.connect(websocket)
    df = None
    window = 20

    try:
        while True:
            data = await websocket.receive_text()

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction
            raw_data = list(json_data.values())

            # Add time stamp to the last received data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            data_processor.add_data(json_data)
            # this line save the recent 100 samples to the CSV file. you can change 100 if you want.
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()

            newDataDf = pd.DataFrame(json_data, index=[0])
            newDataDf["timestamp"] = pd.Timestamp.now()
            newDataDf["label"] = None

            if df is None:
                df = newDataDf
            else:
                df = pd.concat([df, newDataDf], ignore_index=True)
                if len(df) > window:
                    df = df[1:]

            """  
            In this line we use the model to predict the labels.
            Right now it only return 0.
            You need to modify the predict_label function to return the true label
            """

            label = predict_label(model, scaler, df)
            json_data["label"] = current_label or label
            df.loc[df.index[-1], "label"] = current_label or label

            # Calculate the distribution of the last 50 labels
            label_distribution = df["label"].value_counts(normalize=True).to_dict()

            # print the last data in the terminal
            print(json_data)
            print(
                label_distribution  # .to_string(header=["Label", "Proportion"], index=True)
            )
            json_data["confidence"] = label_distribution

            # broadcast the last data to webpage
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
