import matplotlib
matplotlib.use("Agg")

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from src.model import ECGCNN
from src.dataset import ECGDataset
from src.saliency import compute_integrated_gradients
from src.validation import compute_qrs_focus_score
from config import DEVICE


def find_strong_examples(model, dataset):
    normal_example = None
    abnormal_example = None

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for i in indices:
        x, y = dataset[i]
        x_input = x.unsqueeze(0).to(DEVICE)

        output = model(x_input)
        pred = torch.argmax(output, dim=1).item()

        explanation = compute_integrated_gradients(model, x_input, pred, steps=50)
        score = compute_qrs_focus_score(explanation)

        if y.item() == 0 and normal_example is None:
            normal_example = (x, explanation, score)

        if y.item() == 1 and abnormal_example is None:
            abnormal_example = (x, explanation, score)

        if normal_example and abnormal_example:
            break

    return normal_example, abnormal_example


def plot_comparison(normal_data, abnormal_data, filename="ecg_comparison_panel.png"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Normal
    x_n, expl_n, score_n = normal_data
    ecg_n = x_n.detach().squeeze().cpu().numpy()
    expl_n = expl_n / (expl_n.max() + 1e-8)

    axes[0].plot(ecg_n, label="ECG")
    axes[0].plot(expl_n, alpha=0.8, label="Integrated Gradients")
    axes[0].set_title(f"Normal Beat | QRS Focus: {score_n:.4f}")
    axes[0].legend()

    # Abnormal
    x_a, expl_a, score_a = abnormal_data
    ecg_a = x_a.detach().squeeze().cpu().numpy()
    expl_a = expl_a / (expl_a.max() + 1e-8)

    axes[1].plot(ecg_a, label="ECG")
    axes[1].plot(expl_a, alpha=0.8, label="Integrated Gradients")
    axes[1].set_title(f"Abnormal Beat | QRS Focus: {score_a:.4f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    print("Loading dataset...")
    dataset = ECGDataset()

    print("Loading model...")
    model = ECGCNN().to(DEVICE)
    model.load_state_dict(torch.load("ecg_model.pth", map_location=DEVICE))
    model.eval()

    print("Searching for strong examples...")
    normal_data, abnormal_data = find_strong_examples(model, dataset)

    print("Generating comparison panel...")
    filename = "ecg_comparison_panel.png"
    plot_comparison(normal_data, abnormal_data, filename)

    print(f"Saved comparison panel as {filename}")

    # Open automatically on macOS
    os.system(f"open {filename}")


if __name__ == "__main__":
    main()

