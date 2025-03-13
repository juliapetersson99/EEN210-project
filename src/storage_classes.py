from datetime import datetime
import csv
import pandas as pd
import os

class FileStorage:
    def __init__(self, file_path: str = "state_changes.log"):
        self.file_path = file_path

    def save_state_change(self, patient_id: str = "", state: str = "", message: str = "", code: str = ""):
        """Append a state change record to the file.
        
        If some values are missing, they default to empty strings.
        If the file is new or empty, a header row is written.
        """
        timestamp = datetime.now().isoformat()
        # Check if the file exists and is non-empty
        file_exists = os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0
        line = f"{timestamp},{patient_id or ''},{state or ''},{message or ''},{code or ''}\n"
        with open(self.file_path, "a") as f:
            if not file_exists:
                f.write("timestamp,patient_id,state,message,code\n")
            f.write(line)

    def read_all(self):
        """Read all state change records from the file."""
        with open(self.file_path, "r") as f:
            return f.readlines()

    def read_events_for_patient(self, patient_id: str):
        """Read and return all events for the specified patient_id using pandas."""
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            return []
        # Filter the DataFrame for the given patient_id
        df_patient = df[df["patient_id"] == patient_id]
        return df_patient.to_dict(orient="records")
