import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from models import SparseAE_CNN_Mamba  # import from your models.py
import pandas as pd
from data_loader import HTMLDataset
from sklearn.model_selection import KFold
import numpy as np

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
csv_path = 'labeled_unbalanced.csv'
df = pd.read_csv(csv_path)
label_map = {
    'Irrelevant': 0,
    'Attack as a service': 1,
    'Attacker as a service': 2,
    'Malware as a service': 3
}
df['label_numeric'] = df['label'].map(label_map)

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
max_len = 512
n_splits = 5 # Number of folds for cross-validation

# Initialize lists to store metrics for each fold
fold_train_losses = []
fold_val_losses = []
fold_val_accuracies = []

# Prepare the full dataset
print("Loading dataset...")
full_file_paths = df['file'].tolist()
full_numeric_labels = df['label_numeric'].tolist()
full_dataset = HTMLDataset(file_paths=full_file_paths, labels=full_numeric_labels, max_len=max_len)

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Main cross-validation loop with progress bar
fold_progress = tqdm(enumerate(kf.split(full_dataset)), total=n_splits, desc="Cross-Validation Progress", position=0)

for fold, (train_index, val_index) in fold_progress:
    fold_progress.set_description(f"Processing Fold {fold+1}/{n_splits}")
    print(f"\n--- Fold {fold+1}/{n_splits} ---")

    # Create data subsets for the current fold
    train_subset = Subset(full_dataset, train_index)
    val_subset = Subset(full_dataset, val_index)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # -------------------------------
    # Model, Optimizer, Criterion (re-initialized for each fold)
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

    # Lists to store metrics for the current fold
    train_losses, val_losses, val_accuracies = [], [], []

    # -------------------------------
    # Training Loop for the current fold
    # -------------------------------
    epoch_progress = tqdm(range(epochs), desc=f"Fold {fold+1} Epochs", position=1, leave=False)
    
    for epoch in epoch_progress:
        model.train()
        total_loss = 0
        
        # Training progress bar
        train_progress = tqdm(train_loader, 
                            desc=f"Fold {fold+1} Epoch {epoch+1} Training", 
                            position=2, 
                            leave=False)
        
        for inputs, labels in train_progress:
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
            
            # Update training progress bar with current loss
            train_progress.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------------------------------
        # Validation for the current fold
        # -------------------------------
        model.eval()
        total_val_loss, correct, total = 0, 0, 0
        
        # Validation progress bar
        val_progress = tqdm(val_loader, 
                          desc=f"Fold {fold+1} Epoch {epoch+1} Validation", 
                          position=2, 
                          leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_progress:
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
                
                # Update validation progress bar
                current_acc = correct / total if total > 0 else 0
                val_progress.set_postfix({"Val Loss": f"{loss.item():.4f}", "Val Acc": f"{current_acc:.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        # Update epoch progress bar
        epoch_progress.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}",
            "Val Acc": f"{val_accuracy:.4f}"
        })

        print(f"Fold {fold+1} Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_accuracy:.4f}")

    # Store metrics for the current fold
    fold_train_losses.append(train_losses)
    fold_val_losses.append(val_losses)
    fold_val_accuracies.append(val_accuracies)

    # -------------------------------
    # Save checkpoint for the current fold
    # -------------------------------
    checkpoint_path = f"sparse_ae_cnn_mamba_fold{fold+1}_final.pth"
    torch.save(model.state_dict(), checkpoint_path)
    
    # Update fold progress bar
    fold_progress.set_postfix({
        "Completed": f"{fold+1}/{n_splits}",
        "Final Val Acc": f"{val_accuracies[-1]:.4f}"
    })

fold_progress.close()

# -------------------------------
# Average and Plot Results Across Folds
# -------------------------------
print("\nCalculating cross-validation results...")
avg_fold_train_losses = np.mean(fold_train_losses, axis=0)
avg_fold_val_losses = np.mean(fold_val_losses, axis=0)
avg_fold_val_accuracies = np.mean(fold_val_accuracies, axis=0)

print(f"\nCross-Validation Results ({n_splits} folds):")
print(f"Average Training Loss: {avg_fold_train_losses[-1]:.4f}")
print(f"Average Validation Loss: {avg_fold_val_losses[-1]:.4f}")
print(f"Average Validation Accuracy: {avg_fold_val_accuracies[-1]:.4f}")

# Calculate standard deviations for better reporting
final_val_accs = [fold_accs[-1] for fold_accs in fold_val_accuracies]
print(f"Validation Accuracy Std: ±{np.std(final_val_accs):.4f}")

plt.figure(figsize=(12, 6))
plt.plot(avg_fold_train_losses, label="Average Train Loss")
plt.plot(avg_fold_val_losses, label="Average Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Average Training and Validation Loss across {n_splits} Folds")
plt.savefig("avg_loss_plot_cv.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(avg_fold_val_accuracies, label="Average Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title(f"Average Validation Accuracy across {n_splits} Folds")
plt.savefig("avg_accuracy_plot_cv.png")
plt.show()

print("Cross-validation training completed!")