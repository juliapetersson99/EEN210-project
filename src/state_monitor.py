import time
from alert import AlertNotifier


#  codes.
# W19 - Unspecified fall
ALERT_CODES = {
    "fall_notification": "W19 (Fall)",
    "person_safe": "Person recovered",
    "no_movement": "No movement of patient",
    "extended_inactivity": "Activity Reminder"  
}

class FallDetectionStateMonitor:
    def __init__(self, notifier: AlertNotifier, storage = None, patient_id = None):
        self.current_state = "normal"  #initial state
        self.notifier = notifier
        self.last_fall_timestamp = None
        self.inactivity_start = None 
        self.storage = storage
        self.patient_id = patient_id
        self.label_buffer = []  # new buffer for the last 10 labels

    def set_patient_id(self, patient_id):
        self.patient_id = patient_id
    def update_state(self, predicted_label: str):
        current_time = time.time()
        
        # Buffer the latest label and keep maximum 10 entries
        self.label_buffer.append(predicted_label)
        if len(self.label_buffer) > 10:
            self.label_buffer.pop(0)
        # Proceed only if we have 10 labels and they're all identical
        if len(self.label_buffer) < 10 or not all(lbl == self.label_buffer[0] for lbl in self.label_buffer):
            return
        stable_label = self.label_buffer[0]

        #reset inactivity counter if the stable label is different from sitting/standing
        if stable_label not in ["sitting", "standing"]:
            self.inactivity_start = None

        #Fall detection: transition from normal to fall notification.
        if stable_label == "falling" and self.current_state == "normal":
            self.current_state = "fall_notification"
            self.last_fall_timestamp = current_time
            self.inactivity_start = None  # reset inactivity timer
            self.notifier.send_alert(ALERT_CODES["fall_notification"], "Fall detected.")
            if self.storage:
                self.storage.save_state_change(self.patient_id,self.current_state, "Fall detected.", ALERT_CODES["fall_notification"])

        #Patient is laying down after a fall.
        elif stable_label == "laying":
            if self.current_state == "fall_notification":
                # If patient has been laying for more than 10 seconds, alert no movement.
                if current_time - self.last_fall_timestamp >= 10:
                    self.current_state = "no_movement"
                    #no_move_time = current_time - self.last_fall_timestamp
                    self.notifier.send_alert(ALERT_CODES["no_movement"], f"Patient still not moving after fall.")
                    if self.storage:
                        self.storage.save_state_change(self.patient_id,self.current_state, "Patient still not moving after fall.",ALERT_CODES["no_movement"])
            self.inactivity_start = None
            

        
        elif stable_label == "recover" and self.current_state in ["fall_notification", "no_movement"]:
             # If recovering from a fall
            if self.current_state in ["fall_notification", "no_movement"]:
                self.current_state = "person_safe"
                self.notifier.send_alert(ALERT_CODES["person_safe"], "Patient recovered and is safe.")
                if self.storage:
                    self.storage.save_state_change(self.patient_id,self.current_state, "Patient recovered and is safe.",ALERT_CODES["person_safe"])


        elif stable_label in ["sitting", "standing"]:

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
                    self.storage.save_state_change(self.patient_id,self.current_state, "Extended inactivity alert triggered.",ALERT_CODES["extended_inactivity"])
            else:
                if(self.current_state == "person_safe"):
                    self.current_state = "normal"

        # 4. Other activities (e.g., walking) reset inactivity and state set to normal again assumin patient is recovered
        else:
            self.inactivity_start = None
            if stable_label == ['walking','running'] and self.current_state == "person_safe":
                self.current_state = "normal"
                if self.storage:
                    self.storage.save_state_change(self.patient_id,self.current_state, "Patient resumed to activities.")