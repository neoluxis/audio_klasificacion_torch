import argparse
import os
import numpy as np
from collections import Counter, deque
from predict import predict_wav, predict_stream, predict_folder, load_model, CleanAudioDataset


class FilterFactory:
    """Factory to create and manage prediction filters."""
    @staticmethod
    def create_filter(filter_type, n_classes, **kwargs):
        if filter_type == "none":
            return NoneFilter()
        elif filter_type == "kalman":
            process_noise = kwargs.get("process_noise", 0.01)
            measurement_noise = kwargs.get("measurement_noise", 0.1)
            return KalmanFilter(n_classes, process_noise, measurement_noise)
        elif filter_type == "majority":
            window_size = kwargs.get("window_size", 5)
            return MajorityVotingFilter(window_size)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")


class NoneFilter:
    """No-op filter: returns raw predictions."""
    def process(self, probs, class_names):
        pred_idx = np.argmax(probs, axis=1)[0]
        return class_names[pred_idx], probs[0]


class KalmanFilter:
    """Kalman filter for smoothing class probabilities."""
    def __init__(self, n_classes, process_noise=0.01, measurement_noise=0.1):
        self.n_classes = n_classes
        # State: class probabilities [n_classes]
        self.x = np.ones(n_classes) / n_classes  # Initial uniform distribution
        # State covariance
        self.P = np.eye(n_classes)
        # Process noise covariance
        self.Q = np.eye(n_classes) * process_noise
        # Measurement noise covariance
        self.R = np.eye(n_classes) * measurement_noise
        # State transition matrix (identity: assume probabilities evolve slowly)
        self.F = np.eye(n_classes)
        # Measurement matrix (identity: observe probabilities directly)
        self.H = np.eye(n_classes)

    def process(self, probs, class_names):
        probs = probs[0]  # [n_classes]
        # Ensure probs sum to 1 and are non-negative
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= np.sum(probs)

        # Prediction step
        x_pred = self.F @ self.x  # Predicted state
        P_pred = self.F @ self.P @ self.F.T + self.Q  # Predicted covariance

        # Update step
        z = probs  # Measurement
        y = z - self.H @ x_pred  # Measurement residual
        S = self.H @ P_pred @ self.H.T + self.R  # Residual covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S + 1e-8 * np.eye(self.n_classes))  # Kalman gain
        self.x = x_pred + K @ y  # Updated state
        self.P = (np.eye(self.n_classes) - K @ self.H) @ P_pred  # Updated covariance

        # Ensure state is valid probability distribution
        self.x = np.clip(self.x, 1e-8, 1.0)
        self.x /= np.sum(self.x)

        pred_idx = np.argmax(self.x)
        return class_names[pred_idx], self.x


class MajorityVotingFilter:
    """Majority voting filter over a sliding window of predictions."""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)

    def process(self, probs, class_names):
        # Get raw prediction
        pred_idx = np.argmax(probs, axis=1)[0]
        pred_class = class_names[pred_idx]
        # Add to window
        self.predictions.append(pred_class)
        # Majority vote
        if len(self.predictions) == 0:
            return pred_class, probs[0]
        majority_class = Counter(self.predictions).most_common(1)[0][0]
        # Return probabilities of the raw prediction for consistency
        return majority_class, probs[0]


def infer_wav(model, input_name, softmax, wav_path, sample_rate, hop_size, class_names, device, verbose, output_file, filter_obj):
    """Apply filtering to WAV file predictions."""
    # Get raw predictions
    results = predict_wav(model, input_name, softmax, wav_path, sample_rate, hop_size, class_names, device, verbose, output_file=None, output_predictions=False)
    
    # Filter predictions
    filtered_results = []
    for window_idx, (probs, _, time_start, time_end) in enumerate(results):
        pred_class, filtered_probs = filter_obj.process(probs, class_names)
        if verbose:
            prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, filtered_probs)])
            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class} ({prob_str})")
        else:
            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class}")
        
        if output_file:
            with open(output_file, 'a') as f:
                f.write(f"{wav_path}, Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): {pred_class}\n")
        
        filtered_results.append(pred_class)
    
    return filtered_results


def infer_stream(model, input_name, softmax, stream_type, stream_source, input_sr, target_sr, class_names, device, verbose, output_file, filter_obj):
    """Apply filtering to stream predictions."""
    # Get raw predictions
    results = predict_stream(model, input_name, softmax, stream_type, stream_source, input_sr, target_sr, class_names, device, verbose, output_file=None, output_predictions=False)
    
    # Filter predictions
    filtered_results = []
    for window_idx, (probs, _, time_start, time_end) in enumerate(results):
        pred_class, filtered_probs = filter_obj.process(probs, class_names)
        if verbose:
            prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, filtered_probs)])
            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class} ({prob_str})")
        else:
            print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class}")
        
        if output_file:
            with open(output_file, 'a') as f:
                f.write(f"Stream, Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): {pred_class}\n")
        
        filtered_results.append(pred_class)
    
    return filtered_results


def infer_folder(model, input_name, softmax, folder_path, sample_rate, hop_size, class_names, device, verbose, output_file, filter_obj, args):
    """Apply filtering to folder predictions."""
    # Get raw predictions
    all_results = predict_folder(model, input_name, softmax, folder_path, sample_rate, hop_size, class_names, device, verbose, output_file=None, output_predictions=False)
    
    # Check folder structure
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    class_names_set = set(class_names)
    subdirs_set = set(subdirs)
    overlap = len(class_names_set & subdirs_set) / max(len(subdirs), 1)
    
    if overlap >= 0.8 and subdirs:  # Assume dataset structure
        print(f"Detected dataset-like structure in {folder_path}. Evaluating as test set.")
        y_true = []
        y_pred = []
        for wav_path, results in all_results:
            # Reset filter for each WAV file
            filter_obj = FilterFactory.create_filter(args.filter_type, len(class_names),
                                                    window_size=args.filter_window,
                                                    process_noise=args.filter_process_noise,
                                                    measurement_noise=args.filter_measurement_noise)
            # Extract class name from path
            class_name = os.path.basename(os.path.dirname(wav_path))
            if class_name not in class_names:
                print(f"Skipping unknown class file: {wav_path}")
                continue
            
            print(f"\nEvaluating {wav_path} (Ground Truth: {class_name})")
            predictions = []
            for window_idx, (probs, _, time_start, time_end) in enumerate(results):
                pred_class, filtered_probs = filter_obj.process(probs, class_names)
                if verbose:
                    prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, filtered_probs)])
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class} ({prob_str})")
                else:
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class}")
                
                if output_file:
                    with open(output_file, 'a') as f:
                        f.write(f"{wav_path}, Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): {pred_class}\n")
                
                predictions.append(pred_class)
            
            if predictions:
                most_common_pred = Counter(predictions).most_common(1)[0][0]
                y_true.append(class_name)
                y_pred.append(most_common_pred)
                print(f"File Prediction: {most_common_pred} (Ground Truth: {class_name})")
        
        # Compute metrics
        if y_true:
            accuracy = np.mean([y_true[i] == y_pred[i] for i in range(len(y_true))])
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
            print("No valid WAV files found in dataset structure.")
    
    else:  # Non-dataset structure
        print(f"Non-dataset structure detected in {folder_path}. Predicting recursively on all WAV files.")
        for wav_path, results in all_results:
            # Reset filter for each WAV file
            filter_obj = FilterFactory.create_filter(args.filter_type, len(class_names),
                                                    window_size=args.filter_window,
                                                    process_noise=args.filter_process_noise,
                                                    measurement_noise=args.filter_measurement_noise)
            print(f"\nPredicting on {wav_path}")
            for window_idx, (probs, _, time_start, time_end) in enumerate(results):
                pred_class, filtered_probs = filter_obj.process(probs, class_names)
                if verbose:
                    prob_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(class_names, filtered_probs)])
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class} ({prob_str})")
                else:
                    print(f"Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): Predicted: {pred_class}")
                
                if output_file:
                    with open(output_file, 'a') as f:
                        f.write(f"{wav_path}, Window {window_idx} ({time_start:.1f}-{time_end:.1f}s): {pred_class}\n")


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
    
    # Load model
    model, input_name, softmax = load_model(args, device, n_classes)
    
    # Create filter
    filter_obj = FilterFactory.create_filter(args.filter_type, n_classes,
                                            window_size=args.filter_window,
                                            process_noise=args.filter_process_noise,
                                            measurement_noise=args.filter_measurement_noise)
    
    # Predict with filtering
    if args.input_wav:
        if not os.path.exists(args.input_wav):
            raise FileNotFoundError(f"WAV file not found: {args.input_wav}")
        infer_wav(model, input_name, softmax, args.input_wav, args.sample_rate, args.hop_size, class_names, device, args.verbose, args.output_file, filter_obj)
    elif args.input_mic:
        infer_stream(model, input_name, softmax, "mic", args.input_mic, args.input_sr, args.sample_rate, class_names, device, args.verbose, args.output_file, filter_obj)
    elif args.input_stream:
        infer_stream(model, input_name, softmax, "stream", args.input_stream, args.input_sr, args.sample_rate, class_names, device, args.verbose, args.output_file, filter_obj)
    else:  # input_folder
        infer_folder(model, input_name, softmax, args.input_folder, args.sample_rate, args.hop_size, class_names, device, args.verbose, args.output_file, filter_obj, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict audio classification with filtering using PyTorch or ONNX model")
    parser.add_argument("--model_type", type=str, default="conv_rnn",
                        choices=["conv1d", "conv_rnn", "lstm", "transformer"],
                        help="Model type for PyTorch models (default: conv_rnn)")
    parser.add_argument("--model_path", type=str, default="runs/task1/best_model.pth",
                        help="Path to PyTorch (.pth) or ONNX (.onnx) model file")
    parser.add_argument("--dataset_path", type=str, default="./clean",
                        help="Path to dataset directory to infer classes (default: ./clean)")
    parser.add_argument("--input_wav", type=str, default=None,
                        help="Path to input WAV file")
    parser.add_argument("--input_mic", type=str, default=None,
                        help="Microphone device (e.g., hw:2)")
    parser.add_argument("--input_stream", type=str, default="neolux5:40918",
                        help="Server address for audio stream (default: neolux5:40918)")
    parser.add_argument("--input_folder", type=str, default=None,
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
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to output file for saving predictions")
    parser.add_argument("--filter_type", type=str, default="majority",
                        choices=["none", "kalman", "majority"],
                        help="Filter type for post-processing predictions (default: majority)")
    parser.add_argument("--filter_window", type=int, default=5,
                        help="Window size for majority voting filter (default: 5)")
    parser.add_argument("--filter_process_noise", type=float, default=0.01,
                        help="Process noise for Kalman filter (default: 0.01)")
    parser.add_argument("--filter_measurement_noise", type=float, default=0.1,
                        help="Measurement noise for Kalman filter (default: 0.1)")
    
    args = parser.parse_args()
    main(args)