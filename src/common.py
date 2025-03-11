"""Common functions and constants for the project."""

SENSOR_COLS = [
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
]

ADJUST_W_BASELINE = [
    "acceleration_x",
    "acceleration_y",
    "acceleration_z"
]

POSSIBLE_LABELS = [
    "falling",
    "walking",
    "running",
    "sitting",
    "standing",
    "laying",
    "recover",
]
