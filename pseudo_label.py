# pseudo_label.py
import torch, csv
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import HTMLDataset
from models import CNNMambaClassifier, LinearAE_CNN_Mamba, AEMambaClassifier, ConvAutoencoder, SparseAE_CNN_Mamba
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import json

MODEL = "Sparse"
MODEL_PATH = "sparse_ae_cnn_mamba_epoch5.pth"
UNLABELED_DIR = "engpass"
OUTPUT_CSV = "pseudo_labels.csv"
LABELED = "labeled_unbalanced_final.csv"
CONF_THRESHOLD = 0.975
BATCH_SIZE = 16


device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.CrossEntropyLoss()

# 1. Load model
if MODEL == "CNN":
	model = CNNMambaClassifier(
		vocab_size=30522,  # bert-base-uncased
		embed_dim=256,
		cnn_out=128,
		num_classes=4
	)
elif MODEL == "LinAE":
	model = LinearAE_CNN_Mamba(
		vocab_size=30522,   # for BERT tokenizer
		embed_dim=256,
		ae_hidden_dim=1024,
		cnn_out=128,
		num_classes=4,
		seq_len=512
	)
elif MODEL == "Sparse":
    model = SparseAE_CNN_Mamba(
		vocab_size=30522,
		embed_dim=256,
		latent_dim=128,
		cnn_out=128,
		num_classes=4,
		seq_len=512,
		sparsity_weight=1e-3
	).to(device)
else:
	ae = ConvAutoencoder(
		vocab_size=30522,
		embed_dim=256,
		seq_len=512,
		latent_dim=64
	)
	ae.load_state_dict(torch.load("autoencoder.pt"))
	model = AEMambaClassifier(ae, num_classes=4)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# 2. Collect unlabeled files
import os
unlabeled_files = [os.path.join(UNLABELED_DIR, f) for f in os.listdir(UNLABELED_DIR) if f.endswith(".html")]
df = pd.read_csv(LABELED)
file_paths = df['file'].tolist()

unlabeled_files_set = set(unlabeled_files)
file_paths_set = set(file_paths)

unlabeled_files = list(unlabeled_files_set - file_paths_set)

dataset = HTMLDataset(unlabeled_files, labels=None)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# 3. Generate pseudo-labels
all_pseudo_labels_for_json = [] # For the JSON output (all predictions)
high_confidence_pseudo_labels = [] # For the CSV output (only high confidence)

with torch.no_grad():
    for input_ids_batch, paths_batch in tqdm(loader, desc="Generating pseudo-labels"):
        # input_ids_batch is a tensor, paths_batch is a list of strings
        input_ids_batch = input_ids_batch.to(device)

        logits, _, _ = model(input_ids_batch) # only need logits for labeling

        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)

        # Iterate through the predictions for the current batch
        for i in range(len(preds)):
            file_path = paths_batch[i]
            prob = max_probs[i].item()
            pred = preds[i].item()

            # Store all predictions for the JSON file (including low confidence ones as None)
            if prob >= CONF_THRESHOLD:
                all_pseudo_labels_for_json.append({"file": file_path, "label": int(pred), "confidence": float(prob)})
            else:
                all_pseudo_labels_for_json.append({"file": file_path, "label": None, "confidence": float(prob)})

            # Only store high-confidence predictions for the CSV file
            if prob >= CONF_THRESHOLD:
                high_confidence_pseudo_labels.append({"file": file_path, "label": int(pred), "confidence": float(prob)})
# -------------------------------
# Save pseudo-labeled data
# -------------------------------

os.makedirs("pseudo_labels", exist_ok=True)

# Save all predictions (including low confidence ones as None) to JSON
with open("pseudo_labels/unlabeled_with_all_pseudo_labels.json", "w") as f:
    json.dump(all_pseudo_labels_for_json, f, indent=2)

# Save only high-confidence predictions to a CSV file
output_csv_path = os.path.join("pseudo_labels", OUTPUT_CSV)
if high_confidence_pseudo_labels: # Only write if there's data
    df_pseudo = pd.DataFrame(high_confidence_pseudo_labels)
    df_pseudo.to_csv(output_csv_path, index=False)
    print(f"Saved {len(high_confidence_pseudo_labels)} high-confidence pseudo-labels to {output_csv_path}")
else:
    print("No high-confidence pseudo-labels generated to save to CSV.")


print(f"Generated {len(all_pseudo_labels_for_json)} total pseudo-label entries (including low confidence).")
print("High-confidence entries:", sum(1 for x in all_pseudo_labels_for_json if x['label'] is not None))