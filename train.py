import argparse
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import CleanAudioDataset
from model import ModelFactory
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil

from tqdm.rich import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

def get_task_dir(output_dir):
    """Create a unique taskN directory in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    existing_tasks = [d for d in os.listdir(output_dir) if d.startswith("task") and os.path.isdir(os.path.join(output_dir, d))]
    task_nums = [int(d.replace("task", "")) for d in existing_tasks if d.replace("task", "").isdigit()]
    next_task = max(task_nums, default=0) + 1
    task_dir = os.path.join(output_dir, f"task{next_task}")
    os.makedirs(task_dir)
    os.makedirs(os.path.join(task_dir, "tensorboard"))
    return task_dir


def plot_loss_f1(train_losses, val_losses, train_f1s, val_f1s, save_path):
    """Plot and save loss and F1 score curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "b--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("F1 Score", color="tab:orange")
    ax2.plot(epochs, train_f1s, "r-", label="Train F1")
    ax2.plot(epochs, val_f1s, "r--", label="Val F1")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")
    
    plt.title("Training and Validation Loss and F1 Score")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def train_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = CleanAudioDataset(root_dir=args.dataset_path, sample_rate=args.sample_rate, duration=1.0)
    n_classes = len(dataset.get_class_names())
    class_names = dataset.get_class_names()
    
    # Train-validation split
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * len(dataset))
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=2)
    
    # Create model
    model_kwargs = {
        "n_classes": n_classes,
        "sample_rate": args.sample_rate,
        "in_channels": 1,
    }
    if args.model_type in ["conv_rnn", "lstm"]:
        model_kwargs.update({
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers
        })
    if args.model_type == "transformer":
        model_kwargs.update({
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers
        })
    
    model = ModelFactory.create_model(args.model_type, **model_kwargs)
    model = model.to(device)
    
    # Load pretrained model if specified
    if args.pretrained_model is not None:
        if not os.path.exists(args.pretrained_model):
            raise FileNotFoundError(f"Pretrained model file not found: {args.pretrained_model}")
        try:
            pretrained_state = torch.load(args.pretrained_model, map_location=device)
            model.load_state_dict(pretrained_state)
            print(f"Loaded pretrained model from {args.pretrained_model}")
        except Exception as e:
            raise RuntimeError(f"Error loading pretrained model: {str(e)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard writer
    task_dir = get_task_dir(args.output_dir)
    writer = SummaryWriter(os.path.join(task_dir, "tensorboard"))
    
    # Training loop
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    best_val_f1 = -float("inf")
    best_epoch = 0
    
    try:
        for epoch in range(args.epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []
            
            for waveforms, labels in tqdm(train_loader):
                waveforms, labels = waveforms.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_f1 = f1_score(train_labels, train_preds, average="macro")
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            
            with torch.no_grad():
                for waveforms, labels in tqdm(val_loader):
                    waveforms, labels = waveforms.to(device), labels.to(device)
                    outputs = model(waveforms)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_f1 = f1_score(val_labels, val_preds, average="macro")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(task_dir, "best_model.pth"))
                print(f"Saved best model at epoch {best_epoch} with Val F1: {best_val_f1:.4f}")
            
            # Log to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("F1/Train", train_f1, epoch)
            writer.add_scalar("F1/Val", val_f1, epoch)
            
            # Compute confusion matrix
            cm = confusion_matrix(val_labels, val_preds)
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            writer.add_figure("Confusion Matrix", fig, epoch)
            plt.close(fig)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(task_dir, "model.pth"))
    
    # Save plots
    plot_loss_f1(train_losses, val_losses, train_f1s, val_f1s, os.path.join(task_dir, "loss_f1.png"))
    
    # Save final confusion matrix
    final_cm = confusion_matrix(val_labels, val_preds)
    plot_confusion_matrix(final_cm, class_names, os.path.join(task_dir, "confusion_matrix.png"))
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Training complete. Outputs saved in {task_dir}")
    print(f"Best model saved at epoch {best_epoch} with Val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio classification model")
    parser.add_argument("--model_type", type=str, default="resnet_rnn",
                        choices=["conv1d", "conv_rnn", "lstm", "transformer", "resnet", "resnet_rnn"],
                        help="Model type (default: lstm)")
    parser.add_argument("--dataset_path", type=str, default="./clean",
                        help="Path to dataset directory (default: ./clean)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate of audio (default: 16000)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of epochs (default: 10)")
    parser.add_argument("--output_dir", type=str, default="./runs",
                        help="Output directory for task folders (default: ./runs)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for conv_rnn and lstm (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers for conv_rnn, lstm, transformer (default: 2)")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Embedding dimension for transformer (default: 128)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads for transformer (default: 4)")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to pretrained model file (default: None)")
    
    args = parser.parse_args()
    train_model(args)