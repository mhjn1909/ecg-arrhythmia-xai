import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

##############################################
#  DATA PREPROCESSING
##############################################

class ECGDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        X = (X - X.min(axis=1, keepdims=True)) / (
            X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8
        )

        y = np.where(y == 0, 0, 1)

        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


##############################################
# MODEL
##############################################

class ECG_CNN(nn.Module):
    def __init__(self):
        super(ECG_CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),  # fixed input channel
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


##############################################
# TRAINING
##############################################

def train_model(model, train_loader, val_loader, epochs=15):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)}")

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "ecg_cnn_phase1.pth")
    print("Model saved!")


##############################################
# EVALUATION
##############################################

def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()

    all_labels = []
    all_preds = []

    print("Starting evaluation...")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            if batch_idx % 20 == 0:
                print(f"Evaluating batch {batch_idx}/{len(dataloader)}")

    all_labels = np.concatenate(all_labels).ravel()
    all_preds = np.concatenate(all_preds).ravel()

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("Confusion Matrix:\n", cm)

    return cm


##############################################
# CONFUSION MATRIX
##############################################

def plot_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Normal", "Abnormal"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")


##############################################
# MAIN
##############################################

if __name__ == "__main__":

    train_dataset = ECGDataset("data/mitbih_train.csv")
    test_dataset = ECGDataset("data/mitbih_test.csv")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=2)

    model = ECG_CNN().to(DEVICE)

    print("Model initialized successfully.")

    train_model(model, train_loader, test_loader, epochs=15)

    cm = evaluate_model(model, test_loader)
    plot_confusion_matrix(cm)
