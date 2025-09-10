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
from models import MambaClassifier 
from utils import save_plots, save_confusion_matrix 

VOCAB_SIZE = AutoTokenizer.from_pretrained("bert-base-uncased").vocab_size
D_MODEL = 768  # Hidden dimension for Mamba, matches BERT-base-uncased
N_LAYERS = 4 
NUM_CLASSES = 4
MAX_LEN = 512
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
WEIGHT_DECAY = 1e-2 # L2 regularization
GRADIENT_ACCUMULATION_STEPS = 4 # Accumulate gradients over multiple batches
OUTPUT_DIR = "results" # Directory to save plots and models

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

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
    
    optimizer.zero_grad() 

    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        input_ids, labels = input_ids.to(device), labels.to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss = loss / GRADIENT_ACCUMULATION_STEPS 

        loss.backward()
        
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad() 

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS 
        progress_bar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})

    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for input_ids, labels in progress_bar:
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    print(classification_report(all_labels, all_preds, digits=4))
    return avg_loss, accuracy, precision, recall, f1, all_labels, all_preds

def main(csv_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, _ = prepare_data_from_csv(csv_file)

    model = MambaClassifier(
        d_model=D_MODEL, 
        n_layers=N_LAYERS, 
        vocab_size=VOCAB_SIZE, 
        num_classes=NUM_CLASSES, 
        max_len=MAX_LEN
    ).to(device)
    
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    weights = torch.tensor([1.0, 600, 200, 2.0])
    weights = torch.tensor(weights, dtype=torch.float).to(device) 
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    
    train_losses_hist = []
    val_losses_hist = []
    val_accuracies_hist = []
    val_f1_scores_hist = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, all_labels, all_preds = \
            evaluate(model, val_loader, criterion, device)

        train_losses_hist.append(train_loss)
        val_losses_hist.append(val_loss)
        val_accuracies_hist.append(val_accuracy)
        val_f1_scores_hist.append(val_f1)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_mamba_classifier.pth"))
            print(f"  Saved new best model with F1-score: {best_val_f1:.4f}")
            save_confusion_matrix(all_labels, all_preds, LABEL_MAP, output_dir=OUTPUT_DIR, epoch_num=epoch+1)


    print("\nTraining finished!")
    print(f"Best Validation F1-score: {best_val_f1:.4f}")

    print(f"Saving plots to {OUTPUT_DIR}...")
    save_plots(train_losses_hist, val_losses_hist, val_accuracies_hist, val_f1_scores_hist, output_dir=OUTPUT_DIR)
    print("Plots saved successfully.")


if __name__ == "__main__":
    main('final.csv')