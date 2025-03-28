import time
from alert import AlertNotifier
from datetime import datetime

#  codes.
# W19 - Unspecified fall
ALERT_CODES = {
    "fall_notification": "W19 (Fall)",
    "person_safe": "Person recovered",
    "no_movement": "No movement of patient",
    "extended_inactivity": "Activity Reminder",
    "resumed_activity": "Resumed to activities"
}

class FallDetectionStateMonitor:
    def __init__(self, notifier: AlertNotifier, storage = None, patient = None):
        self.current_state = "normal"  #initial state
        self.notifier = notifier
        self.last_fall_timestamp = None
        self.inactivity_start = None 
        self.storage = storage
        self.patient= patient
        self.label_buffer = []  #buffer for the last 10 labels
        self.age = None


    def set_patient(self, patient):
        self.patient = patient
        today = datetime.today()
        try:
            patient_date = datetime.strptime(patient.birthDate.isostring, '%Y-%m-%d')
        except:
            #demo patient
            patient_date = datetime.strptime('1991-01-01', '%Y-%m-%d')
        self.age = today.year - patient_date.year - ((today.month, today.day) < (patient_date.month, patient_date.day))

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

        #needed to do this for demo fix..
        if(self.current_state == "extended_inactivity"):
            self.inactivity_start = None
            self.current_state = "normal"
        #reset inactivity counter if the stable label is different from sitting/standing
        if stable_label not in ["sitting", "standing"]:
            self.inactivity_start = None
        

        #transition from normal to fall notification.
        if stable_label == "falling" and self.current_state == "normal":
            self.current_state = "fall_notification"
            self.last_fall_timestamp = current_time
            self.inactivity_start = None  # reset inactivity timer
            self.notifier.send_alert(ALERT_CODES["fall_notification"], "Fall detected.")
            if self.storage:
                self.storage.save_state_change(self.patient.id,self.current_state, "Fall detected.", ALERT_CODES["fall_notification"])

        #Patient is laying down after a fall.
        elif stable_label == "laying":
            if self.current_state == "fall_notification":
                # If patient has been laying for more than 10 seconds, alert no movement.
                time_limit = 10
                if(self.age >= 70):
                    time_limit = 5

                if current_time - self.last_fall_timestamp >= time_limit:
                    self.current_state = "no_movement"
                    #no_move_time = current_time - self.last_fall_timestamp
                    self.notifier.send_alert(ALERT_CODES["no_movement"], f"Patient still not moving after fall.")
                    if self.storage:
                        self.storage.save_state_change(self.patient.id,self.current_state, "Patient still not moving after fall.",ALERT_CODES["no_movement"])
            self.inactivity_start = None
            

        
        elif stable_label == "recover" and self.current_state in ["fall_notification", "no_movement"]:
             #recovering from a fall
            if self.current_state in ["fall_notification", "no_movement"]:
                self.current_state = "person_safe"
                self.notifier.send_alert(ALERT_CODES["person_safe"], "Patient recovered and is safe.")
                if self.storage:
                    self.storage.save_state_change(self.patient.id,self.current_state, "Patient recovered and is safe.",ALERT_CODES["person_safe"])


        elif stable_label in ["sitting", "standing"] and self.current_state !="extended_inactivity":

            # If transitioning into sitting/standing, record the time.
            if self.inactivity_start is None:
                self.inactivity_start = current_time
            # Check for extended inactivity (adjust as needed, 13 seconds for demo purposes)
            elif current_time - self.inactivity_start >= 13:
                self.current_state = "extended_inactivity"
                self.notifier.send_alert(
                    ALERT_CODES["extended_inactivity"],
                    "Patient has been sitting/standing still for over 30 minutes."
                )
                self.inactivity_start = None
                if self.storage:
                    self.storage.save_state_change(self.patient.id,self.current_state, "Extended inactivity alert triggered.",ALERT_CODES["extended_inactivity"])
            else:
                if(self.current_state == "person_safe"):
                    self.current_state = "normal"


        elif stable_label in ["walking","running"] and self.current_state == "extended_inactivity":
            self.current_state = "normal"
            if self.storage:
                self.storage.save_state_change(self.patient.id,self.current_state, "Patient resumed to activities.",ALERT_CODES["resumed_activity"])
        # reset inactivity timer if patient is walking/running
        else:
            self.inactivity_start = None
            if stable_label == ['walking','running'] and self.current_state == "person_safe":
                self.current_state = "normal"
                if self.storage:
                    self.storage.save_state_change(self.patient.id,self.current_state, "Patient resumed to activities.",ALERT_CODES["resumed_activity"])