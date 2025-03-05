from datetime import datetime

class FileStorage:
    def __init__(self, file_path: str = "state_changes.log"):
        self.file_path = file_path

    def save_state_change(self, patient_id: str, state: str, message: str):
        """Append a state change record to the file including the patient_id."""
        timestamp = datetime.now().isoformat()
        line = f"{timestamp}, {patient_id}, {state}, {message}\n"
        with open(self.file_path, "a") as f:
            f.write(line)

    def read_all(self):
        """Read all state change records from the file."""
        with open(self.file_path, "r") as f:
            return f.readlines()
    def read_events_for_patient(self, patient_id: str):
        """Read and return all events that belong to the specified patient_id."""
        events = []
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    # Expected format: timestamp, patient_id, state, message
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 4:
                        continue  # Skip improperly formatted lines
                    # Since message might contain commas, join all parts after index 2.
                    record = {
                        "timestamp": parts[0],
                        "patient_id": parts[1],
                        "state": parts[2],
                        "message": ", ".join(parts[3:])
                    }
                    if record["patient_id"] == patient_id:
                        events.append(record)
        except FileNotFoundError:
            # If file does not exist, return an empty list.
            return events
        return events