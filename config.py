from pathlib import Path

DATA_DIR = Path("/Users/khalidrazakhan/Desktop/mit-bih-arrhythmia-database-1.0.0")

SAMPLING_RATE = 360
LEAD_INDEX = 0

BEAT_WINDOW_BEFORE = 0.2
BEAT_WINDOW_AFTER = 0.4

DEVICE = "mps"   # If error change to "cpu"
