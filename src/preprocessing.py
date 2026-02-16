import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def highpass_filter(signal, fs, cutoff=0.5):
    b, a = butter(2, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=60.0):
    b, a = iirnotch(freq / (fs / 2), 30)
    return filtfilt(b, a, signal)


def preprocess_ecg(signal, fs):
    signal = highpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = (signal - np.mean(signal)) / np.std(signal)
    return signal

