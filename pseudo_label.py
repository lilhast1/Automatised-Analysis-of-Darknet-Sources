# pseudo_label.py
import torch, csv
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import HTMLDataset
from models import CNNMambaClassifier, LinearAE_CNN_Mamba, AEMambaClassifier, ConvAutoencoder, SparseAE_CNN_Mamba
import torch.nn.functional as F
import tqdm

MODEL = "CNN"
MODEL_PATH = "classifier.pt"
UNLABELED_DIR = "html"
OUTPUT_CSV = "pseudo_labels.csv"
CONF_THRESHOLD = 0.95
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

dataset = HTMLDataset(unlabeled_files, labels=None)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# 3. Generate pseudo-labels
pseudo_labels = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Generating pseudo-labels"):
        batch = batch.to(device)
        logits, _, _ = model(batch)  # only need logits for labeling

        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)

        for prob, pred in zip(max_probs.cpu(), preds.cpu()):
            if prob.item() >= confidence_threshold:
                pseudo_labels.append({"label": int(pred.item()), "confidence": float(prob.item())})
            else:
                pseudo_labels.append({"label": None, "confidence": float(prob.item())})

# -------------------------------
# Save pseudo-labeled data
# -------------------------------
os.makedirs("pseudo_labels", exist_ok=True)
with open("pseudo_labels/unlabeled_with_pseudo_labels.json", "w") as f:
    json.dump(pseudo_labels, f, indent=2)

print(f"Generated {len(pseudo_labels)} pseudo-label entries.")
print("High-confidence entries:", sum(1 for x in pseudo_labels if x['label'] is not None))