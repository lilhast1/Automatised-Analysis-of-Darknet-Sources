import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from transformers import AutoModel

# ---------------------------
# Linear Autoencoder
# ---------------------------
class LinearAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Linear(embed_dim*seq_len, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, embed_dim*seq_len)
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.embed(x).view(x.size(0), -1)           # [B, seq*embed]
        latent = F.relu(self.encoder(x))                # [B, hidden_dim]
        reconstructed = torch.sigmoid(self.decoder(latent)) # [B, seq*embed]
        return reconstructed.view(-1, self.seq_len, self.embed_dim), latent

# ---------------------------
# Convolutional Autoencoder
# ---------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, latent_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, latent_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128, embed_dim, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.embed(x).permute(0,2,1)    # [B, embed, seq]
        latent = self.encoder(x)            # [B, latent, seq//2]
        reconstructed = self.decoder(latent) # [B, embed, seq]
        return reconstructed.permute(0,2,1), latent

# ---------------------------
# CNN + Mamba Classifier
# ---------------------------
class CNNMambaClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, cnn_out, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, cnn_out, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.mamba = Mamba(d_model=cnn_out)
        self.fc = nn.Linear(cnn_out, num_classes)

    def forward(self, x):
        x = self.embed(x).permute(0,2,1)       # [B, embed, seq]
        x = self.pool(F.relu(self.conv(x)))    # [B, cnn_out, seq//2]
        x = x.permute(0,2,1)                   # [B, seq//2, cnn_out]
        x = self.mamba(x).mean(dim=1)
        return self.fc(x)

# ---------------------------
# AE -> Mamba Classifier
# ---------------------------
class AEMambaClassifier(nn.Module):
    def __init__(self, autoencoder, num_classes):
        super().__init__()
        self.autoencoder = autoencoder
        self.mamba = Mamba(d_model=autoencoder.embed.embedding_dim)
        self.fc = nn.Linear(autoencoder.embed.embedding_dim, num_classes)

    def forward(self, x):
        x = self.autoencoder.embed(x)        # [B, seq, embed]
        x = x.permute(0,2,1)                 # [B, embed, seq]
        latent = self.autoencoder.encoder(x) # conv encoder part
        latent = latent.permute(0,2,1)       # [B, seq//2, latent_dim]
        x = self.mamba(latent).mean(dim=1)
        return self.fc(x)

# ---------------------------
# Linear AE -> CNN -> Mamba
# ---------------------------
class LinearAE_CNN_Mamba(nn.Module):
    def __init__(self, vocab_size, embed_dim, ae_hidden_dim, cnn_out, num_classes, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim*seq_len, ae_hidden_dim),
            nn.ReLU(),
            nn.Linear(ae_hidden_dim, embed_dim*seq_len),
            nn.ReLU()
        )
        self.conv = nn.Conv1d(embed_dim, cnn_out, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.mamba = Mamba(d_model=cnn_out)
        self.fc = nn.Linear(cnn_out, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.embed(x).view(B, -1)       # Flatten
        x = self.encoder(x).view(B, self.seq_len, self.embed_dim)
        x = x.permute(0,2,1)                # [B, embed, seq]
        x = self.pool(F.relu(self.conv(x)))
        x = x.permute(0,2,1)
        x = self.mamba(x).mean(dim=1)
        return self.fc(x)

# ---------------------------
# Sparse Linear Autoencoder
# ---------------------------
class SparseLinearAE(nn.Module):
    def __init__(self, embed_dim=256, latent_dim=128, sparsity_weight=1e-3):
        super().__init__()
        self.encoder = nn.Linear(embed_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, embed_dim)
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        # x: (B, L, D)
        latent = torch.relu(self.encoder(x))  # (B, L, latent_dim)
        recon = self.decoder(latent)          # (B, L, D)

        # Sparsity loss (L1 on activations)
        sparsity_loss = self.sparsity_weight * latent.abs().mean()

        return latent, recon, sparsity_loss

# ---------------------------
# Sparse Linear Autoencoder -> CNN -> Mamba
# ---------------------------
class SparseAE_CNN_Mamba(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        embed_dim=256,
        latent_dim=128,
        cnn_out=128,
        num_classes=4,
        seq_len=512,
        sparsity_weight=1e-3
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ae = SparseLinearAE(embed_dim, latent_dim, sparsity_weight)

        self.conv = nn.Conv1d(latent_dim, cnn_out, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(64)

        self.mamba = Mamba(
            d_model=cnn_out,
            d_state=64,
            d_conv=4,
            expand=2
        )

        self.fc = nn.Linear(cnn_out, num_classes)

    def forward(self, x):
        x = self.embed(x)  # (B, L, D)
        latent, recon, sparsity_loss = self.ae(x)

        # CNN expects (B, C, L)
        cnn_in = latent.transpose(1, 2)
        features = self.pool(torch.relu(self.conv(cnn_in)))

        # Mamba expects (B, L, D)
        features = features.transpose(1, 2)
        features = self.mamba(features)

        pooled = features.mean(dim=1)
        out = self.fc(pooled)
        return out, recon, sparsity_loss

# ---------------------------
# Mamba blocks
# ---------------------------
class MambaClassifier(nn.Module):
    def __init__(self, d_model: int, n_layers: int, vocab_size: int, num_classes: int,  max_len: int = 512):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba_layers = nn.ModuleList([Mamba(d_model) for _ in range(n_layers)])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, input_ids):        
        x = self.embedding(input_ids)
        for layer in self.mamba_layers:
            x = layer(x) # (batch_size, sequence_length, d_model)
        x = x.permute(0, 2, 1) 
        x = self.global_pool(x).squeeze(-1) # (batch_size, d_model)
        logits = self.classifier(x) # (batch_size, num_classes)
        return logits


# ---------------------------
# BERT finetune
# ---------------------------
class BERTClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.pooler_output # (batch_size, hidden_size)

        logits = self.classifier(pooled_output) # (batch_size, num_classes)
        return logits