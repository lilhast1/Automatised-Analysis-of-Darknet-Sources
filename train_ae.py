import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_autoencoder(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for x, _ in train_loader:  # labels unused
            x = x.to(device)
            opt.zero_grad()
            recon, _ = model(x)
            target = model.embed(x)         # target is embedding
            loss = criterion(recon, target)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                target = model.embed(x)
                val_loss += criterion(recon, target).item()

        train_losses.append(total_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))

        print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")

    # Plot loss
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.show()

    torch.save(model.state_dict(), "autoencoder.pt")
