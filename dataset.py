import os
from glob import glob
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from preprocess import process_raw, process_mel, process_fft
import matplotlib.pyplot as plt
import librosa.display


class CleanAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=1.0, preprocess="raw"):
        """
        PyTorch Dataset for the cleaned audio dataset produced by clean.py.

        Args:
            root_dir (str): Path to the 'clean' directory (e.g., 'clean').
            sample_rate (int): Expected sample rate (default: 16000 Hz).
            duration (float): Expected audio duration in seconds (default: 1.0).
            preprocess (str): Preprocessing method ("raw", "fft", "mel") (default: "raw").
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.expected_samples = int(sample_rate * duration)
        self.preprocess = preprocess
        
        if preprocess not in ["raw", "fft", "mel"]:
            raise ValueError("preprocess must be one of: 'raw', 'fft', 'mel'")
        
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
        Load an audio file, preprocess it, and return the processed data and label.

        Returns:
            tuple: (processed_data, label)
                - processed_data: Torch tensor of preprocessed data (shape depends on preprocess).
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
            
            # Convert to NumPy for preprocessing
            audio = waveform.numpy()
            
            # Apply preprocessing
            if self.preprocess == "raw":
                processed = process_raw(audio, self.sample_rate)
                processed = torch.from_numpy(processed).float()
            elif self.preprocess == "mel":
                processed = process_mel(audio, self.sample_rate)
                processed = torch.from_numpy(processed).float()
            elif self.preprocess == "fft":
                frq, mag = process_fft(audio, self.sample_rate)
                processed = torch.from_numpy(mag).float()  # Return magnitudes only
            else:
                raise ValueError(f"Unknown preprocess method: {self.preprocess}")
            
            return processed, label
        
        except Exception as e:
            print(f"Error loading {wav_path}: {str(e)}")
            # Return a zero tensor and label to avoid crashing
            if self.preprocess == "mel":
                # Approximate Mel spectrogram shape (n_mels=128, time_frames)
                return torch.zeros(128, 32), label
            elif self.preprocess == "fft":
                return torch.zeros(self.expected_samples // 2), label
            else:  # raw
                return torch.zeros(1, self.expected_samples), label
    
    def get_class_names(self):
        """Return the list of class names."""
        return self.classes
    
    def view(self, idx, method="raw"):
        """
        Visualize a dataset item as a raw waveform, FFT, or Mel spectrogram.
        Always recomputes visualization from the raw waveform for consistency.

        Args:
            idx (int): Index of the dataset item to visualize.
            method (str): Visualization method ("raw", "fft", "mel") (default: "raw").
        """
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Index {idx} out of range. Must be between 0 and {len(self)-1}.")
        
        if method not in ["raw", "fft", "mel"]:
            raise ValueError("method must be one of: 'raw', 'fft', 'mel'")
        
        # Load raw audio waveform
        try:
            wav_path, cls_name = self.audio_files[idx]
            label = self.class_to_idx[cls_name]
            class_name = self.classes[label]
            
            # Load and preprocess raw waveform
            waveform, sr = torchaudio.load(wav_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            current_samples = waveform.shape[1]
            if current_samples < self.expected_samples:
                padding = torch.zeros(1, self.expected_samples - current_samples)
                waveform = torch.cat([waveform, padding], dim=1)
            elif current_samples > self.expected_samples:
                waveform = waveform[:, :self.expected_samples]
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            audio = waveform.numpy().flatten()
            
            print(f"Visualizing sample at index {idx}, Class: {class_name}, Shape: {audio.shape}")
        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}")
            return
        
        # Generate visualization
        if method == "raw":
            plt.figure(figsize=(10, 4))
            plt.plot(np.linspace(0, self.duration, len(audio)), audio)
            plt.title(f'Raw Waveform (Class: {class_name})')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.show()
        
        elif method == "mel":
            mel_spectrogram_db = process_mel(audio, self.sample_rate)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mel_spectrogram_db, sr=self.sample_rate, x_axis='time', y_axis='mel', fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram (Class: {class_name})')
            plt.tight_layout()
            plt.show()
        
        elif method == "fft":
            frq, mag = process_fft(audio, self.sample_rate)
            plt.figure(figsize=(10, 4))
            plt.plot(frq, mag, 'r')
            plt.title(f'FFT of Audio Signal (Class: {class_name})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.grid()
            plt.show()


if __name__ == "__main__":
    # Example usage
    dataset = CleanAudioDataset(root_dir="clean", sample_rate=16000, duration=1.0, preprocess="raw")
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.get_class_names()}")
    
    # Test loading a sample
    processed, label = dataset[0]
    print(f"Sample shape: {processed.shape}, Label: {label}")
    
    # Test visualization
    # dataset.view(0, method="mel")
    # dataset.view(0, method="fft")
    # dataset.view(0, method="raw")
    
    # Example with DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    for batch_processed, batch_labels in dataloader:
        print(f"Batch shapes: {batch_processed.shape}, {batch_labels.shape}")
        break