import argparse
import os
import torch
import torchaudio
import onnxruntime as ort
import numpy as np
import sounddevice as sd
import socket
import csv
from dataset import CleanAudioDataset
from model import ModelFactory
from collections import Counter


def downsample_mono(wav, orig_sr, target_sr=16000):
    """Resample audio to target_sr, convert to mono, normalize, preserve full length."""
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # [1, samples]
    elif wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # Mono
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        wav = resampler(wav)
    wav = wav / (torch.max(torch.abs(wav)) + 1e-8)  # Normalize to [-1, 1]
    return wav.numpy()  # [1, samples]


def prepare_window(window, target_samples=16000):
    """Prepare a single window for model input: pad/trim to target_samples."""
    if window.shape[1] < target_samples:
        window = torch.nn.functional.pad(window, (0, target_samples - window.shape[1]))
    elif window.shape[1] > target_samples:
        window = window[:, :target_samples]
    return window.numpy()  # [1, 16000]


def load_model(args, device, n_classes):
    """Load PyTorch or ONNX model."""
    is_onnx = args.model_path.endswith(".onnx")
    if is_onnx:
        session = ort.InferenceSession(args.model_path)
        input_name = session.get_inputs()[0].name
        return session, input_name, None
    else:
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
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        try:
            state_dict = torch.load(args.model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded PyTorch model from {args.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading PyTorch model: {str(e)}")
        model = model.to(device)
        model.eval()
        return model, None, torch.nn.Softmax(dim=1)


def predict_wav(model, input_name, softmax, wav_path, sample_rate, hop_size, class_names, device, verbose, output_file=None, output_predictions=True):
    """Predict on a WAV file in real-time 1-second windows."""
    try:
        waveform, orig_sr = torchaudio.load(wav_path)
    except Exception as e:
        raise RuntimeError(f"Error loading WAV file {wav_path}: {str(e)}")
    
    waveform = downsample_mono(waveform, orig_sr, sample_rate)  # [1, samples]
    waveform = waveform[0]  # [samples]
    
    window_samples = sample_rate  # 1-second = 16000 samples
    hop_samples = int(hop_size * sample_rate)  # e.g., 1.0s = 16000 samples
    num_samples = len(waveform)
    
    duration = num_samples / sample_rate
    num_windows = max(1, int(np.ceil(num_samples / hop_samples)))
    print(f"Processing WAV file: {wav_path}")
    print(f"Sample Rate: {sample_rate} Hz, Duration: {duration:.1f}s, Samples: {num_samples}, Expected windows: {num_windows}")
    
    results = []
    window_idx = 0
    start = 0
    while start < num_samples:
        end = min(start + window_samples, num_samples)
        window = waveform[start:end]
        window = torch.from_numpy(window).unsqueeze(0)  # [1, samples]
        window = prepare_window(window, window_samples)  # [1, 16000]
        window = window.astype(np.float32).reshape(1, 1, window_samples)  # [1, 1, 16000]
        
        try:
            if input_name:  # ONNX
                outputs = model.run(None, {input_name: window})[0]  # [1, n_classes]
                probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
            else:  # PyTorch
                window = torch.from_numpy(window).to(device)
                with torch.no_grad():
                    outputs = model(window)
                    probs = softmax(outputs).cpu().numpy()
            
            pred_idx = np.argmax(probs, axis=1)[0]
            pred_class = class_names[pred_idx]
            time_start = start / sample_rate
            time_end = end / sample_rate
            
            if output_predictions:
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


def predict_folder(model, input_name, softmax, folder_path, sample_rate, hop_size, class_names, device, verbose, output_file=None, output_predictions=True):
    """Predict on all WAV files in a folder, or evaluate as a test set if structure matches dataset."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Check folder structure
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    class_names_set = set(class_names)
    subdirs_set = set(subdirs)
    overlap = len(class_names_set & subdirs_set) / max(len(subdirs), 1)
    
    if overlap >= 0.8 and subdirs:  # Assume dataset structure if ≥80% subdirs match classes
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
                results = predict_wav(model, input_name, softmax, wav_path, sample_rate, hop_size, class_names, device, verbose, output_file=None, output_predictions=False)
                all_results.append((wav_path, results))
                # Write CSV with ground truth
                if output_file:
                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for window_idx, (_, pred_class, time_start, time_end) in enumerate(results):
                            writer.writerow([wav_path, window_idx, time_start, time_end, pred_class, class_name])
                # Use most frequent prediction for the file
                predictions = [r[1] for r in results]  # pred_class
                if predictions:
                    most_common_pred = Counter(predictions).most_common(1)[0][0]
                    y_true.append(class_name)
                    y_pred.append(most_common_pred)
                    if output_predictions:
                        print(f"File Prediction: {most_common_pred} (Ground Truth: {class_name})")
        
        # Compute metrics
        if y_true:
            accuracy = np.mean([y_true[i] == y_pred[i] for i in range(len(y_true))])
            if output_predictions:
                print(f"\nTest Set Evaluation:")
                print(f"Accuracy: {accuracy:.4f} ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(y_true)})")
                
                # Confusion matrix
                cm = np.zeros((len(class_names), len(class_names)), dtype=int)
                class_to_idx = {name: idx for idx, name in enumerate(class_names)}
                for t, p in zip(y_true, y_pred):
                    cm[class_to_idx[t], class_to_idx[p]] += 1
                
                print("\nConfusion Matrix:")
                print("Rows: Ground Truth, Columns: Predicted")
                header = " " * 20 + " ".join(f"{name[:8]:8}" for name in class_names)
                print(header)
                for i, row in enumerate(cm):
                    row_str = f"{class_names[i][:18]:18} | {' '.join(f'{x:8}' for x in row)}"
                    print(row_str)
        else:
            if output_predictions:
                print("No valid WAV files found in dataset structure.")
        
        return all_results
    
    else:  # Non-dataset structure: predict recursively
        print(f"Non-dataset structure detected in {folder_path}. Predicting recursively on all WAV files.")
        all_results = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    print(f"\nPredicting on {wav_path}")
                    results = predict_wav(model, input_name, softmax, wav_path, sample_rate, hop_size, class_names, device, verbose, output_file, output_predictions)
                    all_results.append((wav_path, results))
        return all_results


def predict_stream(model, input_name, softmax, stream_type, stream_source, input_sr, target_sr, class_names, device, verbose, output_file=None, output_predictions=True):
    """Predict on microphone or TCP stream."""
    chunk_samples = int(input_sr)  # 1-second at input_sr
    buffer = np.zeros(chunk_samples, dtype=np.float32)
    
    if stream_type == "mic":
        stream = sd.InputStream(samplerate=input_sr, channels=1, dtype="float32", blocksize=chunk_samples)
        stream.start()
    else:  # stream
        host, port = stream_source.split(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))
        sock.setblocking(False)  # Non-blocking for robust reading
    
    window_idx = 0
    temp_buffer = []  # Dynamic buffer for TCP stream
    results = []
    try:
        while True:
            if stream_type == "mic":
                data, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("Warning: Audio buffer overflowed")
                buffer = data[:, 0]  # Mono
            else:  # stream
                try:
                    raw_data = sock.recv(8192)  # Smaller chunks to avoid blocking
                    if not raw_data:
                        break
                    data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    temp_buffer.extend(data)
                except BlockingIOError:
                    continue  # No data available yet
                
                while len(temp_buffer) >= chunk_samples:
                    buffer = np.array(temp_buffer[:chunk_samples], dtype=np.float32)
                    temp_buffer = temp_buffer[chunk_samples:]
                    
                    wav = downsample_mono(buffer, input_sr, target_sr)
                    wav = prepare_window(torch.from_numpy(wav), target_samples=target_sr)  # [1, 16000]
                    wav = wav.astype(np.float32).reshape(1, 1, 16000)  # [1, 1, 16000]
                    
                    try:
                        if input_name:  # ONNX
                            outputs = model.run(None, {input_name: wav})[0]
                            probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
                        else:  # PyTorch
                            wav = torch.from_numpy(wav).to(device)
                            with torch.no_grad():
                                outputs = model(wav)
                                probs = softmax(outputs).cpu().numpy()
                        
                        pred_idx = np.argmax(probs, axis=1)[0]
                        pred_class = class_names[pred_idx]
                        time_start = window_idx * 1.0
                        time_end = time_start + 1.0
                        
                        if output_predictions:
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
                
                continue  # Wait for more data if buffer not full
            
            # Microphone processing
            wav = downsample_mono(buffer, input_sr, target_sr)
            wav = prepare_window(torch.from_numpy(wav), target_samples=target_sr)  # [1, 16000]
            wav = wav.astype(np.float32).reshape(1, 1, 16000)  # [1, 1, 16000]
            
            try:
                if input_name:  # ONNX
                    outputs = model.run(None, {input_name: wav})[0]
                    probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
                else:  # PyTorch
                    wav = torch.from_numpy(wav).to(device)
                    with torch.no_grad():
                        outputs = model(wav)
                        probs = softmax(outputs).cpu().numpy()
                
                pred_idx = np.argmax(probs, axis=1)[0]
                pred_class = class_names[pred_idx]
                time_start = window_idx * 1.0
                time_end = time_start + 1.0
                
                if output_predictions:
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
    # Validate inputs
    input_count = sum([args.input_wav is not None, args.input_mic is not None, args.input_stream is not None, args.input_folder is not None])
    if input_count != 1:
        raise ValueError("Exactly one of --input_wav, --input_mic, --input_stream, or --input_folder must be specified")
    
    # Validate hop_size
    if args.hop_size <= 0:
        raise ValueError("hop_size must be greater than 0 seconds")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset to infer n_classes and class names
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_path}")
    try:
        dataset = CleanAudioDataset(root_dir=args.dataset_path, sample_rate=args.sample_rate, duration=1.0)
        n_classes = len(dataset.get_class_names())
        class_names = dataset.get_class_names()
        print(f"Inferred {n_classes} classes from dataset: {class_names}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")
    
    # Create output directory
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize CSV file
    if args.output_file:
        if args.input_folder:
            # Check if folder is a test set
            subdirs = [d for d in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, d))]
            class_names_set = set(class_names)
            subdirs_set = set(subdirs)
            overlap = len(class_names_set & subdirs_set) / max(len(subdirs), 1)
            is_test_set = overlap >= 0.8 and subdirs
            headers = ["File", "Window", "Time Start", "Time End", "Predicted", "Ground Truth"] if is_test_set else ["Source", "Window", "Time Start", "Time End", "Predicted"]
        else:
            headers = ["Source", "Window", "Time Start", "Time End", "Predicted"]
        
        with open(args.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    # Load model
    model, input_name, softmax = load_model(args, device, n_classes)
    
    # Predict based on input source
    if args.input_wav:
        if not os.path.exists(args.input_wav):
            raise FileNotFoundError(f"WAV file not found: {args.input_wav}")
        predict_wav(model, input_name, softmax, args.input_wav, args.sample_rate, args.hop_size, class_names, device, args.verbose, args.output_file)
    elif args.input_mic:
        predict_stream(model, input_name, softmax, "mic", args.input_mic, args.input_sr, args.sample_rate, class_names, device, args.verbose, args.output_file)
    elif args.input_stream:
        predict_stream(model, input_name, softmax, "stream", args.input_stream, args.input_sr, args.sample_rate, class_names, device, args.verbose, args.output_file)
    else:  # input_folder
        predict_folder(model, input_name, softmax, args.input_folder, args.sample_rate, args.hop_size, class_names, device, args.verbose, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict audio classification using PyTorch or ONNX model")
    parser.add_argument("--model_type", type=str, default="conv_rnn",
                        choices=["conv1d", "conv_rnn", "lstm", "transformer", "resnet", "resnet_rnn"],
                        help="Model type for PyTorch models (default: conv_rnn)")
    parser.add_argument("--model_path", type=str, default="runs/task4/best_model.pth",
                        help="Path to PyTorch (.pth) or ONNX (.onnx) model file")
    parser.add_argument("--dataset_path", type=str, default="./clean",
                        help="Path to dataset directory to infer classes (default: ./clean)")
    parser.add_argument("--input_wav", type=str, default=None,
                        help="Path to input WAV file")
    parser.add_argument("--input_mic", type=str, default=None,
                        help="Microphone device (e.g., hw:2)")
    parser.add_argument("--input_stream", type=str, default=None,
                        help="Server address for audio stream")
    parser.add_argument("--input_folder", type=str, default='audio2025_rec',
                        help="Path to folder containing WAV files for prediction or test set evaluation")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate for model (default: 16000)")
    parser.add_argument("--input_sr", type=int, default=44100,
                        help="Input sample rate for mic/stream (default: 44100)")
    parser.add_argument("--hop_size", type=float, default=1.0,
                        help="Hop size in seconds for WAV file sliding window (default: 1.0)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for conv_rnn and lstm (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers for conv_rnn, lstm, transformer (default: 2)")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Embedding dimension for transformer (default: 64)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads for transformer (default: 4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print probability scores for each class")
    parser.add_argument("--output_file", type=str, default="outputs/predict.csv",
                        help="Path to output CSV file for saving predictions (default: outputs/predict.csv)")
    
    args = parser.parse_args()
    main(args)