import wfdb
import numpy as np
from config import BEAT_WINDOW_BEFORE, BEAT_WINDOW_AFTER, SAMPLING_RATE


def load_record(record_path, lead_index):
    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:, lead_index]
    annotation = wfdb.rdann(str(record_path), 'atr')
    return signal, annotation.sample, annotation.symbol


def segment_beats(signal, r_peaks, labels):
    beats = []
    beat_labels = []

    before = int(BEAT_WINDOW_BEFORE * SAMPLING_RATE)
    after = int(BEAT_WINDOW_AFTER * SAMPLING_RATE)

    for r, label in zip(r_peaks, labels):
        start = r - before
        end = r + after

        if start < 0 or end >= len(signal):
            continue

        beat = signal[start:end]

        if np.any(np.isnan(beat)):
            continue

        beats.append(beat)
        beat_labels.append(label)

    return np.array(beats), beat_labels

