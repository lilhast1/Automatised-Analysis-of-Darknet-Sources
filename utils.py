import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def save_plots(train_losses, val_losses, val_accuracies, val_f1_scores, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'o-', label='Training Loss')
    plt.plot(epochs, val_losses, 'o-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()
    
    # F1-score plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1_scores, 'o-', label='Validation F1-score (Weighted)', color='purple')
    plt.title('Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'f1_score_plot.png'))
    plt.close()


def save_confusion_matrix(all_labels, all_preds, label_map, output_dir="results", epoch_num=None):
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    
    id_to_label = {v: k for k, v in label_map.items()}
    class_names = [id_to_label[i] for i in sorted(id_to_label.keys())]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    title = 'Confusion Matrix'
    if epoch_num is not None:
        title += f' (Epoch {epoch_num})'
    plt.title(title)
    plt.tight_layout()
    
    filename = 'confusion_matrix.png'
    if epoch_num is not None:
        filename = f'confusion_matrix_epoch_{epoch_num}.png'

    plt.savefig(os.path.join(output_dir, filename))
    plt.close()