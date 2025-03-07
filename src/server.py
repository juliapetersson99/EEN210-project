import os
import json
import uvicorn
from fastapi import FastAPI, WebSocket, Query, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import time

# Local imports
from alert import ConsoleNotifier
from state_monitor import FallDetectionStateMonitor
from storage_classes import FileStorage
from websocket import WebSocketManager
from process import DataProcessor
from predict import load_model

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
static_dir = os.path.join(current_dir, "static")
html_path = os.path.join(current_dir, "UI_bars.html")
with open(html_path, "r") as f:
    html = f.read()

# Variables for processing data (depends on processing power)
window = 100
send_interval = 6  # Define the interval for sending data
predict_interval = 2

# Initialize file storage (the log file will be created or appended to)
storage = FileStorage(file_path="state_changes.log")
notifier = ConsoleNotifier()
state_monitor = FallDetectionStateMonitor(notifier, storage)

websocket_manager = WebSocketManager()
model, scaler, feature_cols = load_model("final_model")


@app.get("/logs")
def get_logs(patientId: str = Query(...)):
    # Retrieve logs for the given patient
    events = storage.read_events_for_patient(patientId)
    return events


app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/main.css")
async def get_css():
    return FileResponse(os.path.join(static_dir, "main.css"))


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_label, window, send_interval, predict_interval
    patient_id = websocket.query_params.get("patientId", "unknown")

    # TODO: spelling mistake ;)
    state_monitor.set_patinet_id(patient_id)

    data_processor = DataProcessor(
        state_monitor,
        window=window,
        model=model,
        scaler=scaler,
        send_interval=send_interval,
        predict_interval=predict_interval,
        feature_cols=feature_cols,
    )
    await websocket_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()  # Start timing

            json_data = json.loads(data)

            json_response = data_processor.process_data(json_data)

            end_time = time.time()  # End timing
            print(f"Processing time: {end_time-start_time:.3f} seconds")

            if json_response is not None:
                print(json_response)
                await websocket_manager.broadcast_message(json.dumps(json_response))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
