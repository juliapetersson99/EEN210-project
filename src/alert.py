from abc import ABC, abstractmethod
import requests
class AlertNotifier(ABC):
    @abstractmethod
    def send_alert(self, alert_code: str, message: str):
        """Send an alert with a given ICD10C alert code and message."""
        pass

class ConsoleNotifier(AlertNotifier):
    def send_alert(self, alert_code: str, message: str):
        # For demo purposes, simply print the alert.
        print(f"ALERT [{alert_code}]: {message}")
class TelegramNotifier(AlertNotifier):
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
    def send_alert(self, alert_code: str, message: str):
        #just need to create a bot in telegram and get the token and chat id for the channel
        requests.get(
            f"https://api.telegram.org/bot{self.token}/sendMessage?chat_id={self.chat_id}&text=ALERT%20[{alert_code}]:%20{message}"
        )
        #print(f"ALERT [{alert_code}]: {message}")