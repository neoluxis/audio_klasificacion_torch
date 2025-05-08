import argparse
import os
import torch
import onnx
from model import ModelFactory
from dataset import CleanAudioDataset


def export_to_onnx(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset to infer n_classes
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_path}")
    try:
        dataset = CleanAudioDataset(root_dir=args.dataset_path, sample_rate=args.sample_rate, duration=1.0)
        n_classes = len(dataset.get_class_names())
        print(f"Inferred {n_classes} classes from dataset: {dataset.get_class_names()}")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")
    
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
    
    # Load trained model weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {args.model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {str(e)}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input for export
    dummy_input = torch.randn(1, 1, 16000, device=device)  # [batch_size=1, channels=1, samples=16000]
    
    # Determine output path
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.splitext(args.model_path)[0] + ".onnx"
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"Exported ONNX model to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error verifying ONNX model: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model_type", type=str, default="resnet",
                        choices=["conv1d", "conv_rnn", "lstm", "transformer", "resnet", "resnet_rnn"],
                        help="Model type (default: lstm)")
    parser.add_argument("--model_path", type=str, default='runs/task5/best_model.pth',
                        help="Path to trained model file (e.g., runs/task1/best_model.pth)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save ONNX model (default: model_path with .onnx extension)")
    parser.add_argument("--dataset_path", type=str, default="./clean",
                        help="Path to dataset directory to infer number of classes (default: ./clean)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate of audio (default: 16000)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for conv_rnn and lstm (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers for conv_rnn, lstm, transformer (default: 2)")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Embedding dimension for transformer (default: 128)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads for_transformer (default: 4)")
    
    args = parser.parse_args()
    export_to_onnx(args)