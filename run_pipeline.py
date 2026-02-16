# ================= TEAM B =================
from src.preprocessing import preprocess_signal
from src.segmentation import segment_beats
from demo import generate_explainability

# ================= TEAM A =================
import torch
from ecg_cnn_phase1 import ECG_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = ECG_CNN().to(DEVICE)
    model.load_state_dict(
        torch.load("ecg_cnn_phase1.pth", map_location=DEVICE)
    )
    model.eval()
    return model

def main():
    print("Step 1: Preprocessing ECG (Team B)")
    signal, fs = preprocess_signal()

    print("Step 2: Segmenting beats (Team B)")
    beats = segment_beats(signal, fs)
    # beats shape must be (N, 1, 360)

    print("Step 3: Loading trained model (Team A)")
    model = load_model()

    print("Step 4: Predicting")
    beats = torch.tensor(beats, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = torch.sigmoid(model(beats))

    print("Step 5: Explainability (Team B)")
    generate_explainability(model, beats, preds)

    print("✅ PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
