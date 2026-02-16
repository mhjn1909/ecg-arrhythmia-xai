import torch
from torch.utils.data import DataLoader
from src.dataset import ECGDataset
from src.model import ECGCNN
from config import DEVICE

dataset = ECGDataset()
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = ECGCNN().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())

torch.save(model.state_dict(), "ecg_model.pth")

