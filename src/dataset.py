import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

from config import DATA_DIR, LEAD_INDEX, SAMPLING_RATE
from src.preprocessing import preprocess_ecg
from src.segmentation import load_record, segment_beats


class ECGDataset(Dataset):
    def __init__(self):
        self.beats = []
        self.labels = []

        record_files = list(DATA_DIR.glob("*.dat"))

        for record_file in record_files:
            record_name = record_file.stem
            record_path = DATA_DIR / record_name

            signal, r_peaks, symbols = load_record(record_path, LEAD_INDEX)
            signal = preprocess_ecg(signal, SAMPLING_RATE)

            beats, beat_labels = segment_beats(signal, r_peaks, symbols)

            for beat, symbol in zip(beats, beat_labels):
                self.beats.append(beat)
                self.labels.append(self.label_map(symbol))

        self.beats = torch.tensor(np.array(self.beats), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def label_map(self, symbol):
        normal_beats = ['N', 'L', 'R']
        return 0 if symbol in normal_beats else 1

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        return self.beats[idx].unsqueeze(0), self.labels[idx]

