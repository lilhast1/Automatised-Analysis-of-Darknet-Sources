# train.py
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd 
from sklearn.metrics import classification_report
from data_loader import HTMLDataset 
from models import SparseAE_CNN_Mamba 
from utils import save_plots, save_confusion_matrix 

VOCAB_SIZE = AutoTokenizer.from_pretrained("bert-base-uncased").vocab_size
EMBED_DIM = 256        # Embedding dimension for AE
LATENT_DIM = 128       # Latent dimension for AE
CNN_OUT = 128          # Output channels from CNN, also d_model for Mamba
NUM_CLASSES = 4
MAX_LEN = 512          # Sequence length
SPARSITY_WEIGHT = 1e-3 # Sparsity regularization weight
RECON_LOSS_WEIGHT = 1.0 # Weight for the reconstruction loss
CLASSIF_LOSS_WEIGHT = 1.0 # Weight for the classification loss

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-2 
GRADIENT_ACCUMULATION_STEPS = 4 
OUTPUT_DIR = "results_sparse_ae_cnn_mamba"
LABEL_MAP = {
    'Irrelevant': 0,
    'Attack as a service': 1,
    'Attacker as a service': 2,
    'Malware as a service': 3
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_data_from_csv(csv_file_path):
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    if 'file' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'file' and 'label' columns.")

    df['label_id'] = df['label'].map(LABEL_MAP)
    if df['label_id'].isnull().any():
        print("Warning: Some labels in the CSV were not found in LABEL_MAP and will be ignored.")
        df.dropna(subset=['label_id'], inplace=True)
    
    all_file_paths = df['file'].tolist()
    all_labels = df['label_id'].astype(int).tolist()

    if not all_file_paths:
        raise ValueError(f"No valid entries found in {csv_file_path} after mapping labels.")

    valid_paths = []
    valid_labels = []

    for path, label in zip(all_file_paths, all_labels):
        if os.path.exists(path):  # check file existence
            valid_paths.append(path)
            valid_labels.append(label)

    all_file_paths = valid_paths
    all_labels = valid_labels
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_file_paths, all_labels, test_size=0.4, random_state=42, stratify=all_labels
    )
    
    print(f"Total samples: {len(all_file_paths)}")
    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    train_dataset = HTMLDataset(train_paths, train_labels, MAX_LEN)
    val_dataset = HTMLDataset(val_paths, val_labels, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset, val_dataset

def train_epoch(model, dataloader, optimizer, criterion_classif, criterion_recon, device, epoch):
    model.train()
    total_loss = 0
    total_classif_loss = 0
    total_recon_loss = 0
    total_sparsity_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
    
    optimizer.zero_grad() 

    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        input_ids, labels = input_ids.to(device), labels.to(device)

        logits, recon_output, sparsity_loss = model(input_ids)
        
        classif_loss = criterion_classif(logits, labels)
        original_embeddings = model.embed(input_ids).detach() # Detach to prevent gradients flowing back into embedding layer from recon_loss

        recon_loss = criterion_recon(recon_output, original_embeddings) # MSE for reconstruction

        total_batch_loss = (CLASSIF_LOSS_WEIGHT * classif_loss + 
                            RECON_LOSS_WEIGHT * recon_loss + 
                            sparsity_loss) # sparsity_loss already includes its weight
        
        total_batch_loss = total_batch_loss / GRADIENT_ACCUMULATION_STEPS 

        total_batch_loss.backward()
        
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad() 

        total_loss += total_batch_loss.item() * GRADIENT_ACCUMULATION_STEPS
        total_classif_loss += classif_loss.item()
        total_recon_loss += recon_loss.item()
        total_sparsity_loss += sparsity_loss.item() # sparsity_loss is already weighted

        progress_bar.set_postfix({
            'total_loss': f'{total_loss/(batch_idx+1):.4f}',
            'classif_loss': f'{total_classif_loss/(batch_idx+1):.4f}',
            'recon_loss': f'{total_recon_loss/(batch_idx+1):.4f}',
            'sparsity_loss': f'{total_sparsity_loss/(batch_idx+1):.4f}'
        })

    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_total_loss = total_loss / len(dataloader)
    avg_classif_loss = total_classif_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_sparsity_loss = total_sparsity_loss / len(dataloader)

    return avg_total_loss, avg_classif_loss, avg_recon_loss, avg_sparsity_loss

def evaluate(model, dataloader, criterion_classif, criterion_recon, device):
    model.eval()
    total_loss = 0
    total_classif_loss = 0
    total_recon_loss = 0
    total_sparsity_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for input_ids, labels in progress_bar:
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits, recon_output, sparsity_loss = model(input_ids)

            classif_loss = criterion_classif(logits, labels)
            original_embeddings = model.embed(input_ids).detach()
            recon_loss = criterion_recon(recon_output, original_embeddings)
            
            total_batch_loss = (CLASSIF_LOSS_WEIGHT * classif_loss + 
                                RECON_LOSS_WEIGHT * recon_loss + 
                                sparsity_loss)

            total_loss += total_batch_loss.item()
            total_classif_loss += classif_loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_total_loss = total_loss / len(dataloader)
    avg_classif_loss = total_classif_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_sparsity_loss = total_sparsity_loss / len(dataloader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    print(classification_report(all_labels, all_preds, digits=4))
    return avg_total_loss, avg_classif_loss, avg_recon_loss, avg_sparsity_loss, accuracy, precision, recall, f1, all_labels, all_preds

def main(csv_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, _ = prepare_data_from_csv(csv_file)

    model = SparseAE_CNN_Mamba( 
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        cnn_out=CNN_OUT,
        num_classes=NUM_CLASSES,
        seq_len=MAX_LEN,
        sparsity_weight=SPARSITY_WEIGHT
    ).to(device)
    
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    criterion_classif = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss() # MSE for reconstructing embeddings

    best_val_f1 = -1.0
    
    train_total_losses_hist = []
    val_total_losses_hist = []
    val_accuracies_hist = []
    val_f1_scores_hist = []

    for epoch in range(NUM_EPOCHS):
        train_total_loss, train_classif_loss, train_recon_loss, train_sparsity_loss = \
            train_epoch(model, train_loader, optimizer, criterion_classif, criterion_recon, device, epoch)
        
        val_total_loss, val_classif_loss, val_recon_loss, val_sparsity_loss, \
        val_accuracy, val_precision, val_recall, val_f1, all_labels, all_preds = \
            evaluate(model, val_loader, criterion_classif, criterion_recon, device)

        train_total_losses_hist.append(train_total_loss)
        val_total_losses_hist.append(val_total_loss)
        val_accuracies_hist.append(val_accuracy)
        val_f1_scores_hist.append(val_f1)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss (Total): {train_total_loss:.4f} (Classif: {train_classif_loss:.4f}, Recon: {train_recon_loss:.4f}, Sparsity: {train_sparsity_loss:.4f})")
        print(f"  Val Loss (Total): {val_total_loss:.4f} (Classif: {val_classif_loss:.4f}, Recon: {val_recon_loss:.4f}, Sparsity: {val_sparsity_loss:.4f})")
        print(f"  Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_sparse_ae_cnn_mamba_classifier.pth"))
            print(f"  Saved new best model with F1-score: {best_val_f1:.4f}")
            save_confusion_matrix(all_labels, all_preds, LABEL_MAP, output_dir=OUTPUT_DIR, epoch_num=epoch+1)


    print("\nTraining finished!")
    print(f"Best Validation F1-score: {best_val_f1:.4f}")

    print(f"Saving plots to {OUTPUT_DIR}...")
    save_plots(train_total_losses_hist, val_total_losses_hist, val_accuracies_hist, val_f1_scores_hist, output_dir=OUTPUT_DIR)
    print("Plots saved successfully.")


if __name__ == "__main__": 
    main("final.csv")