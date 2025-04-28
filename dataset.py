import os
from glob import glob
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np


class CleanAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=1.0):
        """
        PyTorch Dataset for the cleaned audio dataset produced by clean.py.

        Args:
            root_dir (str): Path to the 'clean' directory (e.g., 'clean').
            sample_rate (int): Expected sample rate (default: 16000 Hz).
            duration (float): Expected audio duration in seconds (default: 1.0).
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.expected_samples = int(sample_rate * duration)
        
        # Get class names and create label mapping
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all WAV files and their labels
        self.audio_files = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            wav_paths = glob(os.path.join(cls_dir, "*.wav"))
            for wav_path in wav_paths:
                self.audio_files.append((wav_path, cls_name))
        
        if not self.audio_files:
            raise ValueError(f"No WAV files found in {root_dir}")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Load an audio file and its label.

        Returns:
            tuple: (audio_tensor, label)
                - audio_tensor: Shape [1, 16000], normalized to [-1, 1].
                - label: Integer class label.
        """
        wav_path, cls_name = self.audio_files[idx]
        label = self.class_to_idx[cls_name]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(wav_path)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Ensure correct length (pad or truncate)
            current_samples = waveform.shape[1]
            if current_samples < self.expected_samples:
                padding = torch.zeros(1, self.expected_samples - current_samples)
                waveform = torch.cat([waveform, padding], dim=1)
            elif current_samples > self.expected_samples:
                waveform = waveform[:, :self.expected_samples]
            
            # Normalize to [-1, 1]
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            return waveform, label
        
        except Exception as e:
            print(f"Error loading {wav_path}: {str(e)}")
            # Return a zero tensor and label to avoid crashing
            return torch.zeros(1, self.expected_samples), label
    
    def get_class_names(self):
        """Return the list of class names."""
        return self.classes


if __name__ == "__main__":
    # Example usage
    dataset = CleanAudioDataset(root_dir="clean", sample_rate=16000, duration=1.0)
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.get_class_names()}")
    
    # Test loading a sample
    waveform, label = dataset[0]
    print(f"Sample shape: {waveform.shape}, Label: {label}")
    
    # Example with DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    for batch_waveforms, batch_labels in dataloader:
        print(f"Batch shapes: {batch_waveforms.shape}, {batch_labels.shape}")
        break