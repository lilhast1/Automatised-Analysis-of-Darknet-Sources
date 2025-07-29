# pseudo_label.py
import torch, csv
from torch.utils.data import DataLoader
from data_loader import HTMLDataset
from models import CNNMambaClassifier, LinearAE_CNN_Mamba, AEMambaClassifier, ConvAutoencoder
import torch.nn.functional as F

MODEL = "CNN"
MODEL_PATH = "classifier.pt"
UNLABELED_DIR = "html"
OUTPUT_CSV = "pseudo_labels.csv"
CONF_THRESHOLD = 0.95
BATCH_SIZE = 16


device = "cuda" if torch.cuda.is_available() else "cpu"

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
    for inputs, _ in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)

        for path, conf, pred in zip(unlabeled_files, max_probs.cpu(), preds.cpu()):
            if conf >= CONF_THRESHOLD:
                pseudo_labels.append((path, pred.item(), conf.item()))

# 4. Save high-confidence pseudo-labels
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "label", "confidence"])
    writer.writerows(pseudo_labels)

print(f"Saved {len(pseudo_labels)} pseudo-labeled files.")
