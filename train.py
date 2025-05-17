import matplotlib
matplotlib.use('Agg')
import argparse
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import CleanAudioDataset
from model import ModelBuilder
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from datetime import datetime
import shutil
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import warnings

# Suppress any experimental warnings if needed
warnings.filterwarnings("ignore")


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def save_model_config(args, task_dir, class_names):
    """Save model configuration as text and YAML."""
    config_dict = {
        "preprocess": args.preprocess,
        "config_path": args.config_path,
        "dataset_path": args.dataset_path,
        "classes": class_names,
        "num_classes": len(class_names),
        "sample_rate": args.sample_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "pretrained_model": args.pretrained_model if args.pretrained_model else "None",
        "seed": args.seed,
        "output_directory": task_dir
    }
    
    # Save as text
    config_text = [
        f"Preprocess: {args.preprocess}",
        f"Config Path: {args.config_path}",
        f"Dataset Path: {args.dataset_path}",
        f"Classes: {', '.join(class_names)}",
        f"Number of Classes: {len(class_names)}",
        f"Sample Rate: {args.sample_rate}",
        f"Batch Size: {args.batch_size}",
        f"Epochs: {args.epochs}",
        f"Learning Rate: {args.lr}",
        f"Pretrained Model: {args.pretrained_model if args.pretrained_model else 'None'}",
        f"Seed: {args.seed}",
        f"Output Directory: {task_dir}"
    ]
    with open(os.path.join(task_dir, "model_config.txt"), "w") as f:
        f.write("\n".join(config_text))
    
    # Save as YAML
    with open(os.path.join(task_dir, "model_config.yaml"), "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False)


def train_model(args):
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    try:
        dataset = CleanAudioDataset(
            root_dir=args.dataset_path,
            sample_rate=args.sample_rate,
            duration=1.0,
            preprocess=args.preprocess
        )
    except ValueError as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")
    
    n_classes = len(dataset.get_class_names())
    class_names = dataset.get_class_names()
    
    # Train-validation split
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(0.8 * len(dataset))
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=0)
    
    # Create model
    model = ModelBuilder.build_model(
        config_path=args.config_path,
        num_classes=n_classes,
        sample_rate=args.sample_rate,
        duration=1.0
    )
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
    
    # Save model configuration
    save_model_config(args, task_dir, class_names)
    
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
            train_batches = 0
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("Batch {task.completed}/{task.total}"),
                TimeRemainingColumn(),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TextColumn("F1: {task.fields[f1]:.4f}")
            ) as progress:
                train_task = progress.add_task(f"Epoch {epoch+1}/{args.epochs} [Train]", total=len(train_loader), loss=0.0, f1=0.0)
                
                for waveforms, labels in train_loader:
                    waveforms, labels = waveforms.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(waveforms)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    train_preds.extend(preds)
                    train_labels.extend(labels.cpu().numpy())
                    
                    # Update running metrics
                    running_train_loss = train_loss / train_batches
                    running_train_f1 = f1_score(train_labels, train_preds, average="macro")
                    progress.update(train_task, advance=1, loss=running_train_loss, f1=running_train_f1)
            
            train_loss /= train_batches
            train_f1 = f1_score(train_labels, train_preds, average="macro")
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            val_batches = 0
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("Batch {task.completed}/{task.total}"),
                TimeRemainingColumn(),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TextColumn("F1: {task.fields[f1]:.4f}")
            ) as progress:
                val_task = progress.add_task(f"Epoch {epoch+1}/{args.epochs} [Val]", total=len(val_loader), loss=0.0, f1=0.0)
                
                with torch.no_grad():
                    for waveforms, labels in val_loader:
                        waveforms, labels = waveforms.to(device), labels.to(device)
                        outputs = model(waveforms)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        val_batches += 1
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_labels.extend(labels.cpu().numpy())
                        
                        # Update running metrics
                        running_val_loss = val_loss / val_batches
                        running_val_f1 = f1_score(val_labels, val_preds, average="macro")
                        progress.update(val_task, advance=1, loss=running_val_loss, f1=running_val_f1)
            
            val_loss /= val_batches
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
            cm = confusion_matrix(val_labels, val_labels)
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            writer.add_figure("Confusion Matrix", fig, epoch)
            plt.close(fig)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            
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
    parser.add_argument("--preprocess", type=str, default="fft",
                        choices=["raw", "fft", "mel"],
                        help="Preprocessing type (default: mel)")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to model YAML config (default: configs/model_<preprocess>.yaml)")
    parser.add_argument("--dataset_path", type=str, default="./clean",
                        help="Path to dataset directory (default: ./clean)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate of audio (default: 16000)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
    parser.add_argument("--output_dir", type=str, default="./runs",
                        help="Output directory for task folders (default: ./runs)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to pretrained model file (default: None)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set default config_path based on preprocess if not provided
    if args.config_path is None:
        args.config_path = f"configs/model_{args.preprocess}.yaml"
    
    train_model(args)