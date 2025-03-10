import time
from alert import AlertNotifier


#  codes.
# W19 - Unspecified fall
ALERT_CODES = {
    "fall_notification": "W19",
    "person_safe": "SAFE",
    "no_movement": "NO_MOVE",
    "extended_inactivity": "EXT_INACT"  
}

class FallDetectionStateMonitor:
    def __init__(self, notifier: AlertNotifier, storage = None, patient_id = None):
        self.current_state = "normal"  #initial state
        self.notifier = notifier
        self.last_fall_timestamp = None
        self.inactivity_start = None 
        self.storage = storage
        self.patient_id = patient_id

    def set_patinet_id(self, patient_id):
        self.patient_id = patient_id
    def update_state(self, predicted_label: str):
        current_time = time.time()

        #reset inactivity counter if the label is different from sitting/standing
        if predicted_label not in ["sitting", "standing"]:
            self.inactivity_start = None

        #Fall detection: transition from normal to fall notification.
        if predicted_label == "falling" and self.current_state == "normal":
            self.current_state = "fall_notification"
            self.last_fall_timestamp = current_time
            self.inactivity_start = None  # reset inactivity timer
            self.notifier.send_alert(ALERT_CODES["fall_notification"], "Fall detected.")
            if self.storage:
                self.storage.save_state_change(self.patient_id,self.current_state, "Fall detected.")

        #Patient is laying down after a fall.
        elif predicted_label == "laying":
            if self.current_state == "fall_notification":
                # If patient has been laying for more than 10 seconds, alert no movement.
                if current_time - self.last_fall_timestamp >= 10:
                    self.current_state = "no_movement"
                    self.notifier.send_alert(ALERT_CODES["no_movement"], "Patient still not moving after fall.")
                    if self.storage:
                        self.storage.save_state_change(self.patient_id,self.current_state, "Patient still not moving after fall.")
            self.inactivity_start = None

        
        elif predicted_label == "recover" and self.current_state in ["fall_notification", "no_movement"]:
             # If recovering from a fall
            if self.current_state in ["fall_notification", "no_movement"]:
                self.current_state = "person_safe"
                self.notifier.send_alert(ALERT_CODES["person_safe"], "Patient recovered and is safe.")
                if self.storage:
                    self.storage.save_state_change(self.patient_id,self.current_state, "Patient recovered and is safe.")


        elif predicted_label in ["sitting", "standing"]:

            # If transitioning into sitting/standing, record the time.
            if self.inactivity_start is None:
                self.inactivity_start = current_time
            # Check for extended inactivity (30 minutes = 1800 seconds)
            elif current_time - self.inactivity_start >= 1800:
                self.current_state = "extended_inactivity"
                self.notifier.send_alert(
                    ALERT_CODES["extended_inactivity"],
                    "Patient has been sitting/standing still for over 30 minutes."
                )
                if self.storage:
                    self.storage.save_state_change(self.patient_id,self.current_state, "Extended inactivity alert triggered.")

        # 4. Other activities (e.g., walking) reset inactivity and state set to normal again assumin patient is recovered
        else:
            self.inactivity_start = None
            if predicted_label == ['walking','running'] and self.current_state == "person_safe":
                self.current_state = "normal"
                if self.storage:
                    self.storage.save_state_change(self.patient_id,self.current_state, "Patient resumed to activities.")