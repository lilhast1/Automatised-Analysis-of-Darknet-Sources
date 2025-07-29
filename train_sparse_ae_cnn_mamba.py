import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models import SparseAE_CNN_Mamba  # import from your models.py

# -------------------------------
# Dummy Dataset Example
# Replace with your HTML text dataset
# -------------------------------
class DummyTextDataset(Dataset):
    def __init__(self, vocab_size=30522, seq_len=512, size=1000, num_classes=4):
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.labels = torch.randint(0, num_classes, (size,))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# -------------------------------
# Hyperparameters
# -------------------------------
vocab_size = 30522
seq_len = 512
num_classes = 4
batch_size = 4      # keep small for 6GB GPU
epochs = 5
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Data Loaders
# -------------------------------
train_dataset = DummyTextDataset(size=5000)
val_dataset   = DummyTextDataset(size=1000)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader    = DataLoader(val_dataset, batch_size=batch_size)

# -------------------------------
# Model, Optimizer, Criterion
# -------------------------------
model = SparseAE_CNN_Mamba(
    vocab_size=vocab_size,
    embed_dim=256,
    latent_dim=128,
    cnn_out=128,
    num_classes=num_classes,
    seq_len=seq_len,
    sparsity_weight=1e-3
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# -------------------------------
# Training Loop
# -------------------------------
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, recon, sparsity_loss = model(inputs)

        # Classification loss
        loss_cls = criterion(logits, labels)

        # Reconstruction loss
        # Use embeddings as target
        emb_target = model.embed(inputs).detach()
        loss_recon = F.mse_loss(recon, emb_target)

        # Total loss
        loss = loss_cls + 0.1*loss_recon + sparsity_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -------------------------------
    # Validation
    # -------------------------------
    model.eval()
    total_val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, recon, sparsity_loss = model(inputs)

            loss_cls = criterion(logits, labels)
            emb_target = model.embed(inputs)
            loss_recon = F.mse_loss(recon, emb_target)
            loss = loss_cls + 0.1*loss_recon + sparsity_loss

            total_val_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_accuracy:.4f}")

    # -------------------------------
    # Save checkpoint
    # -------------------------------
    torch.save(model.state_dict(), f"sparse_ae_cnn_mamba_epoch{epoch+1}.pth")

# -------------------------------
# Plot Loss and Accuracy
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Validation Accuracy")
plt.show()
