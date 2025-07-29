import torch
import matplotlib.pyplot as plt

def train_classifier(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cuda'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)

        train_loss = total_loss/len(train_loader)
        train_acc = correct/total

        # Validation
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item()
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)

        val_loss = total_loss/len(val_loader)
        val_acc = correct/total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Plot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(train_losses, label='train'); plt.plot(val_losses, label='val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(train_accs, label='train'); plt.plot(val_accs, label='val'); plt.title('Accuracy'); plt.legend()
    plt.show()
    plt.savefig('loss_acc.png')
	
    torch.save(model.state_dict(), "classifier.pt")
