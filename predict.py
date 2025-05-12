import argparse
import os
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import socket
import csv
from dataset import CleanAudioDataset
from model import ModelBuilder
from preprocess import process_raw, process_mel, process_fft
from collections import Counter
import yaml
from torch import nn
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def load_model_config(task_dir):
    """Load model configuration from model_config.yaml."""
    config_path = os.path.join(task_dir, "model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def downsample_mono(wav, orig_sr, target_sr=16000):
    """Resample audio to target_sr, convert to mono, return as NumPy."""
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # [1, samples]
    elif wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # Mono
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        wav = resampler(wav)
    return wav.numpy()[0]  # [samples], float32


def prepare_input(audio, preprocess, sample_rate, duration=1.0, n_mels=128):
    """Apply preprocessing using preprocess.py functions."""
    target_samples = int(sample_rate * duration)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    
    if preprocess == "raw":
        processed = process_raw(audio, sample_rate)
        print(f"Raw shape: {processed.shape}, Min: {processed.min():.4f}, Max: {processed.max():.4f}, Mean: {processed.mean():.4f}")
        return torch.from_numpy(processed).float().unsqueeze(0)  # [1, 16000]
    elif preprocess == "fft":
        _, magnitudes = process_fft(audio, sample_rate)
        processed = magnitudes[:target_samples//2]
        print(f"FFT shape: {processed.shape}, Min: {processed.min():.4f}, Max: {processed.max():.4f}, Mean: {processed.mean():.4f}")
        return torch.from_numpy(processed).float().unsqueeze(0)  # [1, 8000]
    elif preprocess == "mel":
        processed = process_mel(audio, sample_rate, n_mels=n_mels)
        print(f"Mel shape: {processed.shape}, Min: {processed.min():.4f}, Max: {processed.max():.4f}, Mean: {processed.mean():.4f}")
        return torch.from_numpy(processed).float().unsqueeze(0)  # [1, n_mels, time_frames]
    else:
        raise ValueError(f"Unsupported preprocess type: {preprocess}")


def load_model(task_dir, model_path, device):
    """Load PyTorch model using ModelBuilder."""
    config = load_model_config(task_dir)
    config_path = config.get("config_path")
    num_classes = config.get("num_classes")
    sample_rate = config.get("sample_rate")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config YAML not found: {config_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = ModelBuilder.build_model(
        config_path=config_path,
        num_classes=num_classes,
        sample_rate=sample_rate,
        duration=1.0
    )
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded PyTorch model from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading PyTorch model: {str(e)}")
    
    model = model.to(device)
    model.eval()
    return model, config


def predict_wav(model, config, wav_path, hop_size, class_names, device, verbose, output_file=None):
    """Predict on a WAV file using 1-second sliding windows."""
    try:
        waveform, orig_sr = torchaudio.load(wav_path)
    except Exception as e:
        raise RuntimeError(f"Error loading WAV file {wav_path}: {str(e)}")
    
    sample_rate = config.get("sample_rate", 16000)
    preprocess = config.get("preprocess", "mel")
    n_mels = config.get("n_mels", 128)
    
    waveform = downsample_mono(waveform, orig_sr, sample_rate)  # [samples]
    num_samples = len(waveform)
    window_samples = sample_rate  # 1-second
    hop_samples = int(hop_size * sample_rate)
    
    duration = num_samples / sample_rate
    num_windows = max(1, int(np.ceil(num_samples / hop_samples)))
    print(f"Processing WAV file: {wav_path}")
    print(f"Sample Rate: {sample_rate} Hz, Duration: {duration:.1f}s, Samples: {num_samples}, Expected windows: {num_windows}")
    
    results = []
    softmax = nn.Softmax(dim=1)
    window_idx = 0
    start = 0
    while start < num_samples:
        end = min(start + window_samples, num_samples)
        window = waveform[start:end]  # [samples]
        window = prepare_input(window, preprocess, sample_rate, n_mels=n_mels)
        
        window = window.to(device)
        try:
            with torch.no_grad():
                outputs = model(window)
                probs = softmax(outputs).cpu().numpy()
            
            pred_idx = np.argmax(probs, axis=1)[0]
            pred_class = class_names[pred_idx]
            time_start = start / sample_rate
            time_end = end / sample_rate
            
            if verbose:
                prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, probs[0])])
                print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class} ({prob_str})")
            else:
                print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class}")
            
            if output_file:
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([wav_path, window_idx, time_start, time_end, pred_class])
            
            results.append((probs, pred_class, time_start, time_end))
            window_idx += 1
            start += hop_samples
        except Exception as e:
            raise RuntimeError(f"Error processing window {window_idx} ({start/sample_rate:.1f}-{end/sample_rate:.1f}s): {str(e)}")
    
    print(f"Processed {window_idx} windows")
    return results


def predict_folder(model, config, folder_path, hop_size, class_names, device, verbose, output_file=None):
    """Predict on all WAV files in a folder or evaluate as a test set."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Check if folder has dataset-like structure
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    class_names_set = set(class_names)
    subdirs_set = set(subdirs)
    overlap = len(class_names_set & subdirs_set) / max(len(subdirs), 1)
    
    if overlap >= 0.8 and subdirs:
        print(f"Detected dataset-like structure in {folder_path}. Evaluating as test set.")
        y_true = []
        y_pred = []
        all_results = []
        for class_name in subdirs:
            if class_name not in class_names:
                print(f"Skipping unknown class directory: {class_name}")
                continue
            class_dir = os.path.join(folder_path, class_name)
            for wav_file in os.listdir(class_dir):
                if not wav_file.lower().endswith('.wav'):
                    continue
                wav_path = os.path.join(class_dir, wav_file)
                print(f"\nEvaluating {wav_path} (Ground Truth: {class_name})")
                results = predict_wav(model, config, wav_path, hop_size, class_names, device, verbose, output_file=None)
                all_results.append((wav_path, results))
                if output_file:
                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for window_idx, (_, pred_class, time_start, time_end) in enumerate(results):
                            writer.writerow([wav_path, window_idx, time_start, time_end, pred_class, class_name])
                predictions = [r[1] for r in results]
                if predictions:
                    most_common_pred = Counter(predictions).most_common(1)[0][0]
                    y_true.append(class_name)
                    y_pred.append(most_common_pred)
                    print(f"File Prediction: {most_common_pred} (Ground Truth: {class_name})")
        
        if y_true:
            accuracy = np.mean([y_true[i] == y_pred[i] for i in range(len(y_true))])
            print(f"\nTest Set Evaluation:")
            print(f"Accuracy: {accuracy:.4f} ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(y_true)})")
            
            cm = np.zeros((len(class_names), len(class_names)), dtype=int)
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            for t, p in zip(y_true, y_pred):
                cm[class_to_idx[t], class_to_idx[p]] += 1
            
            print("\nConfusion Matrix:")
            header = " " * 20 + " ".join(f"{name[:8]:8}" for name in class_names)
            print(header)
            for i, row in enumerate(cm):
                row_str = f"{class_names[i][:18]:18} | {' '.join(f'{x:8}' for x in row)}"
                print(row_str)
        else:
            print("No valid WAV files found in dataset structure.")
        
        return all_results
    
    else:
        print(f"Non-dataset structure detected in {folder_path}. Predicting recursively.")
        all_results = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    print(f"\nPredicting on {wav_path}")
                    results = predict_wav(model, config, wav_path, hop_size, class_names, device, verbose, output_file)
                    all_results.append((wav_path, results))
        return all_results


def predict_stream(model, config, stream_type, stream_source, input_sr, class_names, device, verbose, output_file=None):
    """Predict on microphone or TCP stream."""
    sample_rate = config.get("sample_rate", 16000)
    preprocess = config.get("preprocess", "mel")
    n_mels = config.get("n_mels", 128)
    
    chunk_samples = input_sr  # 1-second at input_sr
    buffer = np.zeros(chunk_samples, dtype=np.float32)
    softmax = nn.Softmax(dim=1)
    
    if stream_type == "mic":
        stream = sd.InputStream(samplerate=input_sr, channels=1, dtype="float32", blocksize=chunk_samples)
        stream.start()
    else:
        host, port = stream_source.split(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))
        sock.setblocking(False)
    
    window_idx = 0
    temp_buffer = []
    results = []
    try:
        while True:
            if stream_type == "mic":
                data, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("Warning: Audio buffer overflowed")
                buffer = data[:, 0]
            else:
                try:
                    raw_data = sock.recv(8192)
                    if not raw_data:
                        break
                    data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    temp_buffer.extend(data)
                except BlockingIOError:
                    continue
                
                while len(temp_buffer) >= chunk_samples:
                    buffer = np.array(temp_buffer[:chunk_samples], dtype=np.float32)
                    temp_buffer = temp_buffer[chunk_samples:]
                    
                    wav = downsample_mono(buffer, input_sr, sample_rate)
                    wav = prepare_input(wav, preprocess, sample_rate, n_mels=n_mels)
                    
                    wav = wav.to(device)
                    try:
                        with torch.no_grad():
                            outputs = model(wav)
                            probs = softmax(outputs).cpu().numpy()
                        
                        pred_idx = np.argmax(probs, axis=1)[0]
                        pred_class = class_names[pred_idx]
                        time_start = window_idx * 1.0
                        time_end = time_start + 1.0
                        
                        if verbose:
                            prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, probs[0])])
                            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class} ({prob_str})")
                        else:
                            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class}")
                        
                        if output_file:
                            with open(output_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([stream_type.capitalize(), window_idx, time_start, time_end, pred_class])
                        
                        results.append((probs, pred_class, time_start, time_end))
                        window_idx += 1
                    except Exception as e:
                        raise RuntimeError(f"Error processing window {window_idx}: {str(e)}")
                
                continue
            
            wav = downsample_mono(buffer, input_sr, sample_rate)
            wav = prepare_input(wav, preprocess, sample_rate, n_mels=n_mels)
            
            wav = wav.to(device)
            try:
                with torch.no_grad():
                    outputs = model(wav)
                    probs = softmax(outputs).cpu().numpy()
                
                pred_idx = np.argmax(probs, axis=1)[0]
                pred_class = class_names[pred_idx]
                time_start = window_idx * 1.0
                time_end = time_start + 1.0
                
                if verbose:
                    prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, probs[0])])
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class} ({prob_str})")
                else:
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class}")
                
                if output_file:
                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([stream_type.capitalize(), window_idx, time_start, time_end, pred_class])
                
                results.append((probs, pred_class, time_start, time_end))
                window_idx += 1
            except Exception as e:
                raise RuntimeError(f"Error processing window {window_idx}: {str(e)}")
            
            if stream_type == "mic":
                buffer = np.zeros(chunk_samples, dtype=np.float32)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if stream_type == "mic":
            stream.stop()
            stream.close()
        else:
            sock.close()
    print(f"Processed {window_idx} windows")
    return results


def main(args):
    """Main function to handle prediction based on input source."""
    input_count = sum([args.input_wav is not None, args.input_mic is not None, args.input_stream is not None, args.input_folder is not None])
    if input_count != 1:
        raise ValueError("Exactly one of --input_wav, --input_mic, --input_stream, or --input_folder must be specified")
    
    if args.hop_size <= 0:
        raise ValueError("hop_size must be greater than 0 seconds")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.task_dir):
        raise FileNotFoundError(f"Task directory not found: {args.task_dir}")
    
    try:
        dataset = CleanAudioDataset(root_dir=args.dataset_path, sample_rate=16000, duration=1.0, preprocess="mel")
        class_names = dataset.get_class_names()
        print(f"Inferred {len(class_names)} classes from dataset: {class_names}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")
    
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        headers = ["Source", "Window", "Time Start", "Time End", "Predicted"]
        if args.input_folder:
            subdirs = [d for d in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, d))]
            class_names_set = set(class_names)
            subdirs_set = set(subdirs)
            overlap = len(class_names_set & subdirs_set) / max(len(subdirs), 1)
            if overlap >= 0.8 and subdirs:
                headers.append("Ground Truth")
        
        with open(args.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    model, config = load_model(args.task_dir, args.model_path, device)
    
    if args.input_wav:
        if not os.path.exists(args.input_wav):
            raise FileNotFoundError(f"WAV file not found: {args.input_wav}")
        predict_wav(model, config, args.input_wav, args.hop_size, class_names, device, args.verbose, args.output_file)
    elif args.input_mic:
        predict_stream(model, config, "mic", args.input_mic, args.input_sr, class_names, device, args.verbose, args.output_file)
    elif args.input_stream:
        predict_stream(model, config, "stream", args.input_stream, args.input_sr, class_names, device, args.verbose, args.output_file)
    else:
        predict_folder(model, config, args.input_folder, args.hop_size, class_names, device, args.verbose, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict audio classification using trained PyTorch model")
    parser.add_argument("--task_dir", type=str, default="./runs/task2",
                        help="Path to task directory containing model_config.yaml (default: ./runs/task1)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to PyTorch model file (.pth). Defaults to task_dir/best_model.pth")
    parser.add_argument("--dataset_path", type=str, default="./clean",
                        help="Path to dataset directory to infer classes (default: ./clean)")
    parser.add_argument("--input_wav", type=str, default=None,
                        help="Path to input WAV file")
    parser.add_argument("--input_mic", type=str, default=None,
                        help="Microphone device (e.g., hw:2)")
    parser.add_argument("--input_stream", type=str, default='neolux5:40918',
                        help="Server address for audio stream (e.g., neolux5:40918)")
    # parser.add_argument("--input_stream", type=str, default=None,
    #                     help="Server address for audio stream (e.g., neolux5:40918)")
    parser.add_argument("--input_folder", type=str, default=None,
                        help="Path to folder containing WAV files for prediction or test set evaluation")
    parser.add_argument("--input_sr", type=int, default=44100,
                        help="Input sample rate for mic/stream (default: 44100)")
    parser.add_argument("--hop_size", type=float, default=1.0,
                        help="Hop size in seconds for WAV file sliding window (default: 1.0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print probability scores for each class")
    parser.add_argument("--output_file", type=str, default="outputs/predict.csv",
                        help="Path to output CSV file for predictions (default: outputs/predict.csv)")
    
    args = parser.parse_args()
    
    # Set default model_path if not provided
    if args.model_path is None:
        args.model_path = os.path.join(args.task_dir, "best_model.pth")
    
    # Test mode: predict on a sample WAV file
    if not any([args.input_wav, args.input_mic, args.input_stream, args.input_folder]):
        print("No input specified. Running test mode with sample WAV file.")
        args.input_wav = "./audio2025_rec/Bearded Seal/Bearded Seal.wav"  # Replace with actual path
        if not os.path.exists(args.input_wav):
            print(f"Sample WAV file {args.input_wav} not found. Please specify an input.")
        else:
            main(args)
    else:
        main(args)