import os
import json
import uvicorn
from fastapi import FastAPI, WebSocket, Query, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
import time
from fhirclient import client
from fhir import fetch_conditions, fetch_medications, demo_patient

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

# Smart Setup

# Initialize the FHIR client

SMART_BASE_URL = "https://gw.interop.community/testjulle/data"
SMART_APP_ID = "340a7665-fc24-4db8-9f90-dc3e06eddc90"
smart_settings = {
    "app_id": SMART_APP_ID,
    "client_id": SMART_APP_ID,
    "api_base": SMART_BASE_URL,
    "aud": SMART_BASE_URL,
    "redirect_uri": "http://localhost:8000/callback",
    "scope": "launch openid profile patient/*.rs",
}
smart = client.FHIRClient(settings=smart_settings)
smart_access_token = None
patient = None

current_dir = os.path.dirname(os.path.abspath(__file__))

template_dir = os.path.join(current_dir, "templates")


templates = Jinja2Templates(directory=template_dir)

# Variables for processing data (depends on processing power)
window = 20
send_interval = 6  # Define the interval for sending data
predict_interval = 2

# Initialize file storage (the log file will be created or appended to)
storage = FileStorage(file_path="state_changes.log")
notifier = ConsoleNotifier()
state_monitor = FallDetectionStateMonitor(notifier, storage)

websocket_manager = WebSocketManager()
model, scaler, feature_cols = load_model("final_model")


@app.get("/")
async def get(request: Request):
    if patient:
        # Fetch patient data
        patient.conditions = fetch_conditions(smart, patient)
        patient.medications = fetch_medications(smart, patient)


    # use demo patient if no real patient is available
    displayPatient = patient or demo_patient()
    state_monitor.set_patient(patient)

    print(displayPatient.as_json())

    return templates.TemplateResponse(
        name="dashboard.html",
        request=request,
        context={
            "patient": displayPatient,
            "access_token": smart_access_token,
            "base_url": SMART_BASE_URL,
        },
    )


@app.get("/launch")
async def launch(request: Request):
    global smart
    # Authorize the user
    launch = request.query_params.get("launch")
    if not launch:
        return "No launch token provided. Please log in with the EHR system."

    # Create a new SMART client with the launch token
    new_settings = {"launch_token": launch, **smart_settings}
    smart = client.FHIRClient(settings=new_settings)

    auth_url = smart.authorize_url
    return RedirectResponse(url=auth_url)


# Callback url for the OAuth2 authorization
@app.get("/callback")
async def callback(request: Request):
    global smart_access_token, patient
    # Extract code from query parameters
    code = request.query_params.get("code")
    if not code:
        return "Authorization failed"

    # Handle token exachange
    smart.handle_callback(str(request.url))

    # Store Access Token
    smart_access_token = smart.server.auth.access_token
    # Fetch patient data
    patient = smart.patient

    return RedirectResponse(url="/")


@app.get("/logs")
def get_logs(patientId: str = Query(...)):
    # Retrieve logs for the given patient
    events = storage.read_events_for_patient(patientId)
    #print(events)
    return events


@app.get("/main.css")
async def get_css():
    return FileResponse(os.path.join(static_dir, "main.css"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_label, window, send_interval, predict_interval

    # print(patient.id)

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
