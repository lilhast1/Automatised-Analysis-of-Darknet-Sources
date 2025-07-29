import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class CNN_MambaClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # CNN layer to extract local n-gram features
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Mamba Layer
        # `d_model` must match the output channels of CNN
        self.mamba = Mamba(d_model=128)

        # Final classification head
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]

        x = self.pool(F.relu(self.conv1(x)))  # [batch_size, 128, reduced_seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, reduced_seq_len, 128]

        # Mamba expects input shape [batch, seq, dim]
        x = self.mamba(x)  # [batch_size, reduced_seq_len, 128]

        # Use mean pooling across time dimension
        x = x.mean(dim=1)  # [batch_size, 128]

        return self.fc(x)  # [batch_size, num_classes]

class AE_MambaClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim, num_classes, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Load pretrained encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, latent_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.encoder.load_state_dict(torch.load('encoder.pt'))

        # Mamba layer expects [batch, seq, dim], so we transpose
        self.mamba = Mamba(d_model=latent_dim)

        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)             # [batch, seq, embed_dim]
        x = x.permute(0, 2, 1)            # [batch, embed_dim, seq]
        x = self.encoder(x)               # [batch, latent_dim, reduced_seq]
        x = x.permute(0, 2, 1)            # [batch, reduced_seq, latent_dim]
        x = self.mamba(x)                 # [batch, reduced_seq, latent_dim]
        x = x.mean(dim=1)                 # [batch, latent_dim]
        return self.fc(x)                 # [batch, num_classes]

class LinearAE_CNN_Mamba(nn.Module):
    def __init__(self, vocab_size, embed_dim, ae_hidden_dim, cnn_out_channels, num_classes, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Step 1: Token Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Step 2: Linear Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * seq_len, ae_hidden_dim),
            nn.ReLU(),
            nn.Linear(ae_hidden_dim, embed_dim * seq_len),
            nn.ReLU()
        )
        self.encoder.load_state_dict(torch.load('encoder.pt'))
        # Step 3: CNN expects (B, C_in, Seq)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_out_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Step 4: Mamba (on reduced seq_len)
        self.mamba = Mamba(d_model=cnn_out_channels)

        # Step 5: Classifier
        self.fc = nn.Linear(cnn_out_channels, num_classes)

    def forward(self, x):  # x: [batch, seq_len]
        B = x.size(0)

        # [B, seq_len, embed_dim]
        x = self.embedding(x)

        # Flatten to [B, seq_len * embed_dim] → Linear AE
        x = x.view(B, -1)
        x = self.encoder(x)  # [B, seq_len * embed_dim]

        # Reshape back to [B, seq_len, embed_dim]
        x = x.view(B, self.seq_len, self.embed_dim)
        x = x.permute(0, 2, 1)  # [B, embed_dim, seq_len] for Conv1d

        x = self.pool(torch.relu(self.conv(x)))  # [B, cnn_out, reduced_seq_len]
        x = x.permute(0, 2, 1)  # [B, reduced_seq_len, cnn_out]

        x = self.mamba(x)  # [B, reduced_seq_len, cnn_out]

        x = x.mean(dim=1)  # [B, cnn_out]
        return self.fc(x)  # [B, num_classes]
