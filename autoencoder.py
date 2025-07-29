import torch
import torch.nn as nn
import torch.nn.functional as F

class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, latent_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder (mirrors encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128, embed_dim, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, x):  # x = token_ids: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch, embed_dim, seq_len]

        latent = self.encoder(x)            # [batch, latent_dim, reduced_seq_len]
        reconstructed = self.decoder(latent)  # [batch, embed_dim, seq_len]

        return reconstructed, latent

# Create dummy autoencoder model for pretraining
class AEOnly(nn.Module):
    def __init__(self, vocab_size, embed_dim, ae_hidden_dim, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * seq_len, ae_hidden_dim),
            nn.ReLU(),
            nn.Linear(ae_hidden_dim, embed_dim * seq_len),
            nn.ReLU()
        )
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, x):
        B = x.size(0)
        embedded = self.embedding(x).view(B, -1)
        reconstructed = self.encoder(embedded)
        return reconstructed.view(B, self.seq_len, self.embed_dim), self.embedding(x)

