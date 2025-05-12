import torch
import torch.nn as nn
import yaml
import math


def get_input_shape(preprocess, sample_rate, duration, n_mels=128, hop_length=512):
    """Compute input shape based on preprocessing type."""
    if preprocess == "raw":
        return int(sample_rate * duration)
    elif preprocess == "fft":
        return int((sample_rate * duration) // 2)
    elif preprocess == "mel":
        # Number of time frames = ceil((sample_rate * duration) / hop_length)
        time_frames = math.ceil((sample_rate * duration) / hop_length)
        return (n_mels, time_frames)
    else:
        raise ValueError(f"Unsupported preprocess type: {preprocess}")


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool_kernel=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_kernel) if pool_kernel else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
    
    def forward(self, x):
        # x: [batch, channels, height, width]
        batch, channels, height, width = x.size()
        x = x.view(batch, channels * height, width).transpose(1, 2)  # [batch, width, channels * height]
        x, _ = self.gru(x)
        return x[:, -1, :]  # Return last time step: [batch, hidden_size]


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU() if out_features != 1 else nn.Identity()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class AudioClassifier(nn.Module):
    def __init__(self, input_type, blocks, num_classes):
        super().__init__()
        self.input_type = input_type
        self.blocks = nn.ModuleList(blocks)
        self.final_fc = nn.Linear(blocks[-1].fc.out_features, num_classes) if isinstance(blocks[-1], FCBlock) else nn.Linear(blocks[-1].gru.hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_type in ["mel", "raw", "fft"]:
            x = x.unsqueeze(1)  # Add channel dim: [batch, 1, ...]
        for block in self.blocks:
            x = block(x)
        return self.final_fc(x)


class ModelBuilder:
    @staticmethod
    def build_model(config_path, num_classes, sample_rate=16000, duration=1.0):
        """Build model from YAML config with automatic channel and input inference."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        preprocess = config.get('preprocess', 'mel')
        blocks_config = config.get('blocks', [])
        n_mels = config.get('n_mels', 128)
        hop_length = config.get('hop_length', 512)
        
        # Infer input shape
        input_shape = get_input_shape(preprocess, sample_rate, duration, n_mels, hop_length)
        if preprocess == "mel" and not isinstance(input_shape, tuple):
            raise ValueError(f"Expected tuple for mel input shape, got {input_shape}")
        elif preprocess in ["raw", "fft"] and not isinstance(input_shape, int):
            raise ValueError(f"Expected int for {preprocess} input shape, got {input_shape}")
        
        # Initialize blocks
        blocks = []
        in_channels = 1  # Initial input has 1 channel after unsqueeze
        last_out_features = None
        
        for block in blocks_config:
            block_type = block.get('type')
            
            if block_type == 'cnn':
                initial_channels = block.get('initial_channels', 16)
                num_cnn_layers = block.get('num_cnn_layers', 3)
                kernel_size = block.get('kernel_size', 3)
                stride = block.get('stride', 1)
                padding = block.get('padding', 1)
                pool_kernel = block.get('pool_kernel', 2)
                
                # Automatic channel doubling
                out_channels = initial_channels
                for i in range(num_cnn_layers):
                    cnn_block = CNNBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        pool_kernel=pool_kernel
                    )
                    blocks.append(cnn_block)
                    in_channels = out_channels
                    out_channels *= 2  # Double channels for next layer
                last_out_features = in_channels
            
            elif block_type == 'gru':
                hidden_size = block.get('hidden_size', 128)
                num_layers = block.get('num_layers', 1)
                dropout = block.get('dropout', 0.0)
                
                # Compute input_size for GRU
                if preprocess == "mel" and blocks:
                    # Simulate CNN output shape to compute GRU input_size
                    x = torch.randn(1, 1, n_mels, input_shape[1])
                    for b in blocks:
                        x = b(x)
                    input_size = x.size(1) * x.size(2)  # channels * height
                else:
                    input_size = last_out_features or input_shape
                gru_block = GRUBlock(input_size, hidden_size, num_layers, dropout)
                blocks.append(gru_block)
                last_out_features = hidden_size
            
            elif block_type == 'fc':
                in_features = last_out_features or input_shape
                out_features = block.get('out_features', 128)
                dropout = block.get('dropout', 0.0)
                fc_block = FCBlock(in_features, out_features, dropout)
                blocks.append(fc_block)
                last_out_features = out_features
        
        return AudioClassifier(preprocess, blocks, num_classes)


if __name__ == "__main__":
    config_path = "configs/model_mel.yaml"
    sample_rate = 16000
    duration = 1.0
    num_classes = 10  # Arbitrary for testing
    batch_size = 2
    
    # Load config and get input shape
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocess = config.get('preprocess')
    n_mels = config.get('n_mels', 128)
    hop_length = config.get('hop_length', 512)
    
    if preprocess != "mel":
        raise ValueError(f"Expected 'mel' preprocessing in {config_path}, got {preprocess}")
    
    input_shape = get_input_shape(preprocess, sample_rate, duration, n_mels, hop_length)
    print(f"Input shape for {preprocess}: {input_shape}")
    
    # Create dummy input
    input_tensor = torch.randn(batch_size, input_shape[0], input_shape[1])  # [batch, n_mels, time_frames]
    
    # Build model
    model = ModelBuilder.build_model(
        config_path=config_path,
        num_classes=num_classes,
        sample_rate=sample_rate,
        duration=duration
    )
    
    # Forward pass
    try:
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: [batch_size={batch_size}, num_classes={num_classes}]")
        assert output.shape == (batch_size, num_classes), f"Output shape mismatch: got {output.shape}, expected {(batch_size, num_classes)}"
        print("Model test passed successfully!")
    except Exception as e:
        print(f"Model test failed: {str(e)}")