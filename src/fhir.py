from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse
import requests
import uvicorn
from fhirclient import client
import os
import jwt
from urllib.parse import urlencode


app = FastAPI()

# Initialize the FHIR client
settings = {
    "app_id": "aa8e20c9-ee1c-4660-bf7b-814b7be41f6b",
    "client_id": "aa8e20c9-ee1c-4660-bf7b-814b7be41f6b",
    "api_base": "https://gw.interop.community/testjulle/data",
    "aud": "https://gw.interop.community/testjulle/data",
    "redirect_uri": "http://localhost:8080/callback",
    "scope": "launch openid profile patient/*.read",
}
smart = client.FHIRClient(settings=settings)
smart.scope = settings["scope"]


@app.get("/authorize")
async def authorize():
    # Redirect the user to the authorization server
    auth_url = smart.authorize_url
    return RedirectResponse(url=auth_url)


@app.get("/index", response_class=HTMLResponse)
async def homepage(launch: str = None):
    print(launch)
    return """
    <html>
        <head>
            <title>FHIR Patient Data</title>
        </head>
        <body>
            <h1>Welcome to the FHIR Patient Data App</h1>
            <p>Please <a href="/authorize">authorize</a> to view patient data.</p>
        </body>
    </html>
    """


# Step 1: Launch endpoint (EHR will redirect here with the launch token)
@app.get("/")
async def launch(request: Request):
    global smart
    launch = request.query_params.get("launch")
    """
    The EHR will redirect to this endpoint with the 'launch' token.
    """
    try:
        # Decode and validate the launch token (JWT)
        # decoded_token = jwt.decode(launch, options={"verify_signature": False})

        # Validate 'aud' (audience) and 'iss' (issuer) fields from the decoded token
        # if decoded_token["aud"] != CLIENT_ID:
        #     return {"error": "Invalid audience (client_id)"}
        # if decoded_token["iss"] != FHIR_SERVER_URL:
        #     return {"error": "Invalid issuer"}

        # Store the launch token in the session or some persistent storage if necessary
        # Redirect user to the authorization server
        scopes = "launch openid fhirUser profile patient/*.rs"  # Correct scope format

        # Step 1: Use the launch token for authorization
        new_settings = {"launch_token": launch, **settings}

        # Create a new SMART client
        smart = client.FHIRClient(settings=new_settings)

        # Store state for later verification
        # You might want to use a session or database here

        # Construct the authorization URL with the launch parameter
        auth_url = smart.authorize_url
        print(auth_url)

        # Redirect the user to the authorization server
        return RedirectResponse(url=auth_url)

    except jwt.ExpiredSignatureError:
        return {"error": "Launch token has expired"}
    except Exception as e:
        return {"error": f"Error decoding launch token: {e}"}


@app.get("/callback", response_class=HTMLResponse)
async def callback(request: Request):
    # Extract code from query parameters
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    print(state)
    if not code:
        return HTMLResponse(
            """
        <html>
            <head>
                <title>Error</title>
            </head>
            <body>
                <h1>Error</h1>
                <p>No authorization code provided.</p>
            </body>
        </html>
        """,
            status_code=400,
        )

    # Handle the callback and exchange the authorization code for an access token
    print(code)
    smart.handle_callback(str(request.url))
    patient = smart.patient
    if patient:
        # Fetch patient information
        patient_data = patient.as_json()
        patient_info = f"""
        <html>
            <head>
                <title>Patient Data</title>
            </head>
            <body>
                <h1>Patient Information</h1>
                <pre>{patient_data}</pre>
            </body>
        </html>
        """
        return patient_info
    else:
        return {"error": "No patient information available"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
