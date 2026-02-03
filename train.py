import torch
from torch.utils.data import DataLoader, random_split
from dataset import DentalDataset
from model import get_model
import segmentation_models_pytorch as smp
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DentalDataset("../data/images", "../data/masks")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

model = get_model().to(device)

dice = smp.losses.DiceLoss(mode="binary")
bce = torch.nn.BCEWithLogitsLoss()

def loss_fn(p, t):
    return dice(p, t) + bce(p, t)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

os.makedirs("../outputs/models", exist_ok=True)

for epoch in range(40):
    model.train()
    total_loss = 0

    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        loss = loss_fn(p, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "../outputs/models/best_model.pth")
