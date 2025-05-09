import argparse
import os
import numpy as np
from collections import Counter, deque
from predict import predict_wav, predict_stream, predict_folder, load_model, CleanAudioDataset
import torch
import csv


class FilterFactory:
    """Factory to create and manage prediction filters."""
    @staticmethod
    def create_filter(filter_type, n_classes, **kwargs):
        if filter_type == "none":
            return NoneFilter()
        elif filter_type == "kalman":
            process_noise = kwargs.get("process_noise", 0.0005)  # Stronger smoothing
            measurement_noise = kwargs.get("measurement_noise", 0.005)
            return KalmanFilter(n_classes, process_noise, measurement_noise)
        elif filter_type == "majority":
            window_size = kwargs.get("window_size", 3)
            return MajorityVotingFilter(window_size)
        elif filter_type == "exponential_moving_average":
            alpha = kwargs.get("alpha", 0.7)  # More responsive
            return ExponentialMovingAverageFilter(n_classes, alpha)
        elif filter_type == "hybrid":
            window_size = kwargs.get("window_size", 3)
            process_noise = kwargs.get("process_noise", 0.0005)
            measurement_noise = kwargs.get("measurement_noise", 0.005)
            return HybridFilter(n_classes, window_size, process_noise, measurement_noise)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")


class NoneFilter:
    """No-op filter: returns raw predictions."""
    def process(self, probs, class_names, confidence_threshold=0.7):
        probs = probs[0]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        if max_prob < confidence_threshold:
            return "uncertain", probs
        return class_names[pred_idx], probs


class KalmanFilter:
    """Kalman filter for smoothing class probabilities."""
    def __init__(self, n_classes, process_noise=0.0005, measurement_noise=0.005):
        self.n_classes = n_classes
        self.x = np.ones(n_classes) / n_classes
        self.P = np.eye(n_classes)
        self.Q = np.eye(n_classes) * process_noise
        self.R = np.eye(n_classes) * measurement_noise
        self.F = np.eye(n_classes)
        self.H = np.eye(n_classes)

    def process(self, probs, class_names, confidence_threshold=0.7):
        probs = probs[0]
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= np.sum(probs)

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        z = probs
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S + 1e-8 * np.eye(self.n_classes))
        self.x = x_pred + K @ y
        self.P = (np.eye(self.n_classes) - K @ self.H) @ P_pred

        self.x = np.clip(self.x, 1e-8, 1.0)
        self.x /= np.sum(self.x)

        max_prob = np.max(self.x)
        pred_idx = np.argmax(self.x)
        if max_prob < confidence_threshold:
            return "uncertain", self.x
        return class_names[pred_idx], self.x


class MajorityVotingFilter:
    """Majority voting filter over a sliding window of predictions."""
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)

    def process(self, probs, class_names, confidence_threshold=0.7):
        probs = probs[0]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        pred_class = "uncertain" if max_prob < confidence_threshold else class_names[pred_idx]
        self.predictions.append(pred_class)
        majority_class = Counter(self.predictions).most_common(1)[0][0] if self.predictions else pred_class
        return majority_class, probs


class ExponentialMovingAverageFilter:
    """Exponential moving average filter for smoothing class probabilities."""
    def __init__(self, n_classes, alpha=0.7):
        self.n_classes = n_classes
        self.alpha = alpha
        self.smoothed_probs = np.ones(n_classes) / n_classes

    def process(self, probs, class_names, confidence_threshold=0.7):
        probs = probs[0]
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= np.sum(probs)
        
        self.smoothed_probs = self.alpha * probs + (1 - self.alpha) * self.smoothed_probs
        self.smoothed_probs = np.clip(self.smoothed_probs, 1e-8, 1.0)
        self.smoothed_probs /= np.sum(self.smoothed_probs)
        
        max_prob = np.max(self.smoothed_probs)
        pred_idx = np.argmax(self.smoothed_probs)
        if max_prob < confidence_threshold:
            return "uncertain", self.smoothed_probs
        return class_names[pred_idx], self.smoothed_probs


class HybridFilter:
    """Hybrid filter combining majority voting and Kalman smoothing."""
    def __init__(self, n_classes, window_size=3, process_noise=0.0005, measurement_noise=0.005):
        self.kalman = KalmanFilter(n_classes, process_noise, measurement_noise)
        self.majority = MajorityVotingFilter(window_size)
        self.n_classes = n_classes

    def process(self, probs, class_names, confidence_threshold=0.7):
        # Apply Kalman smoothing
        kalman_class, kalman_probs = self.kalman.process(probs, class_names, confidence_threshold)
        # Apply majority voting on Kalman predictions
        self.majority.predictions.append(kalman_class)
        majority_class = Counter(self.majority.predictions).most_common(1)[0][0] if self.majority.predictions else kalman_class
        return majority_class, kalman_probs


def infer_wav(model, input_name, softmax, wav_path, sample_rate, hop_size, class_names, device, verbose, output_file, filter_obj, confidence_threshold):
    results = predict_wav(model, input_name, softmax, wav_path, sample_rate, hop_size, class_names, device, verbose, output_file=None, output_predictions=False)
    
    filtered_results = []
    for window_idx, (probs, _, time_start, time_end) in enumerate(results):
        raw_pred = class_names[np.argmax(probs, axis=1)[0]]
        pred_class, filtered_probs = filter_obj.process(probs, class_names, confidence_threshold)
        if verbose:
            prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, filtered_probs)])
            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Raw: {raw_pred}, Filtered: {pred_class} ({prob_str})")
        else:
            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Raw: {raw_pred}, Filtered: {pred_class}")
        
        if output_file:
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([wav_path, window_idx, time_start, time_end, raw_pred, pred_class, *filtered_probs])
        
        filtered_results.append(pred_class)
    
    return filtered_results


def infer_stream(model, input_name, softmax, stream_type, stream_source, input_sr, target_sr, class_names, device, verbose, output_file, filter_obj, confidence_threshold):
    results = predict_stream(model, input_name, softmax, stream_type, stream_source, input_sr, target_sr, class_names, device, verbose, output_file=None, output_predictions=False)
    
    filtered_results = []
    for window_idx, (probs, _, time_start, time_end) in enumerate(results):
        raw_pred = class_names[np.argmax(probs, axis=1)[0]]
        pred_class, filtered_probs = filter_obj.process(probs, class_names, confidence_threshold)
        if verbose:
            prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, filtered_probs)])
            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Raw: {raw_pred}, Filtered: {pred_class} ({prob_str})")
        else:
            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Raw: {raw_pred}, Filtered: {pred_class}")
        
        if output_file:
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([stream_type.capitalize(), window_idx, time_start, time_end, raw_pred, pred_class, *filtered_probs])
        
        filtered_results.append(pred_class)
    
    return filtered_results


def infer_folder(model, input_name, softmax, folder_path, sample_rate, hop_size, class_names, device, verbose, output_file, filter_obj, args):
    all_results = predict_folder(model, input_name, softmax, folder_path, sample_rate, hop_size, class_names, device, verbose, output_file=None, output_predictions=False)
    
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    class_names_set = set(class_names)
    subdirs_set = set(subdirs)
    overlap = len(class_names_set & subdirs_set) / max(len(subdirs), 1)
    
    if overlap >= 0.8 and subdirs:
        print(f"Detected dataset-like structure in {folder_path}. Evaluating as test set.")
        y_true = []
        y_pred_raw = []
        y_pred_filtered = []
        for wav_path, results in all_results:
            filter_obj = FilterFactory.create_filter(args.filter_type, len(class_names),
                                                    window_size=args.filter_window,
                                                    process_noise=args.filter_process_noise,
                                                    measurement_noise=args.filter_measurement_noise,
                                                    alpha=args.filter_alpha)
            class_name = os.path.basename(os.path.dirname(wav_path))
            if class_name not in class_names:
                print(f"Skipping unknown class file: {wav_path}")
                continue
            
            print(f"\nEvaluating {wav_path} (Ground Truth: {class_name})")
            predictions_raw = []
            predictions_filtered = []
            for window_idx, (probs, _, time_start, time_end) in enumerate(results):
                raw_pred = class_names[np.argmax(probs, axis=1)[0]]
                pred_class, filtered_probs = filter_obj.process(probs, class_names, args.confidence_threshold)
                if verbose:
                    prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, filtered_probs)])
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Raw: {raw_pred}, Filtered: {pred_class} ({prob_str})")
                else:
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Raw: {raw_pred}, Filtered: {pred_class}")
                
                if output_file:
                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([wav_path, window_idx, time_start, time_end, raw_pred, pred_class, class_name, *filtered_probs])
                
                predictions_raw.append(raw_pred)
                predictions_filtered.append(pred_class)
            
            if predictions_filtered:
                most_common_raw = Counter([p for p in predictions_raw if p != "uncertain"]).most_common(1)[0][0] if [p for p in predictions_raw if p != "uncertain"] else "uncertain"
                most_common_filtered = Counter([p for p in predictions_filtered if p != "uncertain"]).most_common(1)[0][0] if [p for p in predictions_filtered if p != "uncertain"] else "uncertain"
                y_true.append(class_name)
                y_pred_raw.append(most_common_raw)
                y_pred_filtered.append(most_common_filtered)
                print(f"File Prediction: Raw: {most_common_raw}, Filtered: {most_common_filtered} (Ground Truth: {class_name})")
        
        if y_true:
            accuracy_raw = np.mean([y_true[i] == y_pred_raw[i] for i in range(len(y_true)) if y_pred_raw[i] != "uncertain"])
            accuracy_filtered = np.mean([y_true[i] == y_pred_filtered[i] for i in range(len(y_true)) if y_pred_filtered[i] != "uncertain"])
            print(f"\nTest Set Evaluation:")
            print(f"Raw Accuracy: {accuracy_raw:.4f} ({sum(1 for t, p in zip(y_true, y_pred_raw) if t == p and p != 'uncertain')}/{len([p for p in y_pred_raw if p != 'uncertain'])})")
            print(f"Filtered Accuracy: {accuracy_filtered:.4f} ({sum(1 for t, p in zip(y_true, y_pred_filtered) if t == p and p != 'uncertain')}/{len([p for p in y_pred_filtered if p != 'uncertain'])})")
            
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            cm_raw = np.zeros((len(class_names), len(class_names)), dtype=int)
            cm_filtered = np.zeros((len(class_names), len(class_names)), dtype=int)
            for t, p in zip(y_true, y_pred_raw):
                if p != "uncertain":
                    cm_raw[class_to_idx[t], class_to_idx[p]] += 1
            for t, p in zip(y_true, y_pred_filtered):
                if p != "uncertain":
                    cm_filtered[class_to_idx[t], class_to_idx[p]] += 1
            
            print("\nRaw Confusion Matrix:")
            print("Rows: Ground Truth, Columns: Predicted")
            header = " " * 20 + " ".join(f"{name[:8]:8}" for name in class_names)
            print(header)
            for i, row in enumerate(cm_raw):
                row_str = f"{class_names[i][:18]:18} | {' '.join(f'{x:8}' for x in row)}"
                print(row_str)
            
            print("\nFiltered Confusion Matrix:")
            print("Rows: Ground Truth, Columns: Predicted")
            print(header)
            for i, row in enumerate(cm_filtered):
                row_str = f"{class_names[i][:18]:18} | {' '.join(f'{x:8}' for x in row)}"
                print(row_str)
        else:
            print("No valid WAV files found in dataset structure.")
    
    else:
        print(f"Non-dataset structure detected in {folder_path}. Predicting recursively on all WAV files.")
        for wav_path, results in all_results:
            filter_obj = FilterFactory.create_filter(args.filter_type, len(class_names),
                                                    window_size=args.filter_window,
                                                    process_noise=args.filter_process_noise,
                                                    measurement_noise=args.filter_measurement_noise,
                                                    alpha=args.filter_alpha)
            print(f"\nPredicting on {wav_path}")
            for window_idx, (probs, _, time_start, time_end) in enumerate(results):
                raw_pred = class_names[np.argmax(probs, axis=1)[0]]
                pred_class, filtered_probs = filter_obj.process(probs, class_names, args.confidence_threshold)
                if verbose:
                    prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, filtered_probs)])
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Raw: {raw_pred}, Filtered: {pred_class} ({prob_str})")
                else:
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Raw: {raw_pred}, Filtered: {pred_class}")
                
                if output_file:
                    with open(output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([wav_path, window_idx, time_start, time_end, raw_pred, pred_class, *filtered_probs])


def main(args):
    input_count = sum([args.input_wav is not None, args.input_mic is not None, args.input_stream is not None, args.input_folder is not None])
    if input_count != 1:
        raise ValueError("Exactly one of --input_wav, --input_mic, --input_stream, or --input_folder must be specified")
    
    if args.hop_size <= 0:
        raise ValueError("hop_size must be greater than 0 seconds")
    
    if args.filter_alpha <= 0 or args.filter_alpha >= 1:
        raise ValueError("filter_alpha must be between 0 and 1")
    
    if args.confidence_threshold < 0 or args.confidence_threshold > 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_path}")
    try:
        dataset = CleanAudioDataset(root_dir=args.dataset_path, sample_rate=args.sample_rate, duration=1.0)
        n_classes = len(dataset.get_class_names())
        class_names = dataset.get_class_names()
        print(f"Inferred {n_classes} classes from dataset: {class_names}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")
    
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    if args.output_file:
        if args.input_folder:
            subdirs = [d for d in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, d))]
            class_names_set = set(class_names)
            subdirs_set = set(subdirs)
            overlap = len(class_names_set & subdirs_set) / max(len(subdirs), 1)
            is_test_set = overlap >= 0.8 and subdirs
            headers = ["File", "Window", "Time Start", "Time End", "Raw Predicted", "Filtered Predicted", "Ground Truth"] + [f"Prob_{name}" for name in class_names] if is_test_set else ["Source", "Window", "Time Start", "Time End", "Raw Predicted", "Filtered Predicted"] + [f"Prob_{name}" for name in class_names]
        else:
            headers = ["Source", "Window", "Time Start", "Time End", "Raw Predicted", "Filtered Predicted"] + [f"Prob_{name}" for name in class_names]
        
        with open(args.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    model, input_name, softmax = load_model(args, device, n_classes)
    
    filter_obj = FilterFactory.create_filter(args.filter_type, n_classes,
                                            window_size=args.filter_window,
                                            process_noise=args.filter_process_noise,
                                            measurement_noise=args.filter_measurement_noise,
                                            alpha=args.filter_alpha)
    
    if args.input_wav:
        if not os.path.exists(args.input_wav):
            raise FileNotFoundError(f"WAV file not found: {args.input_wav}")
        infer_wav(model, input_name, softmax, args.input_wav, args.sample_rate, args.hop_size, class_names, device, args.verbose, args.output_file, filter_obj, args.confidence_threshold)
    elif args.input_mic:
        infer_stream(model, input_name, softmax, "mic", args.input_mic, args.input_sr, args.sample_rate, class_names, device, args.verbose, args.output_file, filter_obj, args.confidence_threshold)
    elif args.input_stream:
        infer_stream(model, input_name, softmax, "stream", args.input_stream, args.input_sr, args.sample_rate, class_names, device, args.verbose, args.output_file, filter_obj, args.confidence_threshold)
    else:
        infer_folder(model, input_name, softmax, args.input_folder, args.sample_rate, args.hop_size, class_names, device, args.verbose, args.output_file, filter_obj, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict audio classification with filtering using PyTorch or ONNX model")
    parser.add_argument("--model_type", type=str, default="resnet",
                        choices=["conv1d", "conv_rnn", "lstm", "transformer", "resnet", "resnet_rnn", "sincnet"],
                        help="Model type for PyTorch models (default: resnet)")
    parser.add_argument("--model_path", type=str, default="runs/task1/best_model.pth",
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
                        help="Hidden size for conv_rnn, lstm, and resnet_rnn (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers for conv_rnn, lstm, transformer, resnet_rnn (default: 2)")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Embedding dimension for transformer (default: 64)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads for transformer (default: 4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print probability scores for each class")
    parser.add_argument("--output_file", type=str, default="outputs/inference.csv",
                        help="Path to output CSV file for saving predictions (default: outputs/inference.csv)")
    parser.add_argument("--filter_type", type=str, default="hybrid",
                        choices=["none", "kalman", "majority", "exponential_moving_average", "hybrid"],
                        help="Filter type for post-processing predictions (default: hybrid)")
    parser.add_argument("--filter_window", type=int, default=3,
                        help="Window size for majority voting filter (default: 3)")
    parser.add_argument("--filter_process_noise", type=float, default=0.0005,
                        help="Process noise for Kalman filter (default: 0.0005)")
    parser.add_argument("--filter_measurement_noise", type=float, default=0.005,
                        help="Measurement noise for Kalman filter (default: 0.005)")
    parser.add_argument("--filter_alpha", type=float, default=0.7,
                        help="Smoothing factor for exponential moving average filter (default: 0.7)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Minimum confidence for predictions (default: 0.7)")
    
    args = parser.parse_args()
    main(args)