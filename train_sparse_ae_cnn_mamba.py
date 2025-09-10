import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models import SparseAE_CNN_Mamba  
import pandas as pd
from data_loader import HTMLDataset
from sklearn.model_selection import train_test_split

csv_path = 'labeled_unbalanced_final.csv' 
df = pd.read_csv(csv_path)
label_map = {
    'Irrelevant': 0,
    'Attack as a service': 1,
    'Attacker as a service': 2,
    'Malware as a service': 3
}

df['label_numeric'] = df['label'].map(label_map)
file_paths = df['file'].tolist()
numeric_labels = df['label_numeric'].tolist()

train_df, val_df = train_test_split(
    df,
    test_size=0.6,
    random_state=42, 
    stratify=df['label_numeric']
)

train_paths = train_df['file'].tolist()
train_labels = train_df['label_numeric'].tolist()

val_paths = val_df['file'].tolist()
val_labels = val_df['label_numeric'].tolist()

# -------------------------------
# Hyperparameters
# -------------------------------
vocab_size = 30522
seq_len = 512
num_classes = 4
batch_size = 4      # 6G VRAM
epochs = 5
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 512
# -------------------------------
# Data Loaders
# -------------------------------

train_dataset = HTMLDataset(file_paths=train_paths, labels=train_labels, max_len=max_len)
val_dataset   = HTMLDataset(file_paths=val_paths, labels=val_labels, max_len=max_len)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

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

        loss_cls = criterion(logits, labels)

        emb_target = model.embed(inputs).detach()
        loss_recon = F.mse_loss(recon, emb_target)

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
    # Save epoch
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
plt.savefig("loss_plot.png") 
plt.show()


plt.figure(figsize=(10,5))
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Validation Accuracy")
plt.savefig("accuracy_plot.png")
plt.show()
