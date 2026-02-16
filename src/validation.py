import numpy as np
from config import SAMPLING_RATE, BEAT_WINDOW_BEFORE


def compute_qrs_focus_score(explanation):
    """
    Measures how much attribution lies inside QRS region.
    """

    beat_length = len(explanation)

    # R-peak index
    r_index = int(BEAT_WINDOW_BEFORE * SAMPLING_RATE)

    # Define QRS window ±30 samples (~80 ms)
    qrs_start = max(r_index - 30, 0)
    qrs_end = min(r_index + 30, beat_length)

    total_importance = np.sum(explanation)
    qrs_importance = np.sum(explanation[qrs_start:qrs_end])

    if total_importance == 0:
        return 0.0

    return qrs_importance / total_importance

