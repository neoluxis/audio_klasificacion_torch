import torch
import torch.nn as nn
import yaml
from typing import Dict, List, Any, Optional

# --- Block Definitions ---
class CNNBlock(nn.Module):
    """Convolutional block with optional pooling."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool: Optional[str] = None,
        pool_kernel: Optional[int] = None,
        is_1d: bool = False
    ):
        super(CNNBlock, self).__init__()
        conv = nn.Conv1d if is_1d else nn.Conv2d
        bn = nn.BatchNorm1d if is_1d else nn.BatchNorm2d
        pool_layer = nn.MaxPool1d if is_1d else nn.MaxPool2d
        
        layers = [
            conv(in_channels, out_channels, kernel_size, stride, padding),
            bn(out_channels),
            nn.ReLU()
        ]
        
        if pool == "max" and pool_kernel:
            layers.append(pool_layer(kernel_size=pool_kernel, stride=pool_kernel))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNetBlock(nn.Module):
    """Residual block with two convolutions and a shortcut."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        is_1d: bool = False
    ):
        super(ResNetBlock, self).__init__()
        conv = nn.Conv1d if is_1d else nn.Conv2d
        bn = nn.BatchNorm1d if is_1d else nn.BatchNorm2d
        
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = bn(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                bn(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class GRUBlock(nn.Module):
    """GRU block for sequential data."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0
    ):
        super(GRUBlock, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 2D [batch, features], add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_size]
        # x: [batch, seq_length, input_size]
        out, _ = self.gru(x)
        # Output: [batch, seq_length, hidden_size]
        # Return last time step: [batch, hidden_size]
        return out.squeeze(1)  # [batch, hidden_size]


class LSTMBlock(nn.Module):
    """LSTM block for sequential data."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0
    ):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return out.squeeze(1)


class FCBlock(nn.Module):
    """Fully connected block with dropout."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.5
    ):
        super(FCBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class AdaptivePoolBlock(nn.Module):
    """Adaptive pooling block for 1D or 2D inputs."""
    def __init__(self, output_size: List[int], is_1d: bool = False):
        super(AdaptivePoolBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size[0]) if is_1d else nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class FlattenBlock(nn.Module):
    """Flatten block to convert to 1D tensor."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


# --- Factory Pattern ---
class BlockFactory:
    """Factory to create blocks based on configuration."""
    @staticmethod
    def create_block(config: Dict[str, Any]) -> nn.Module:
        block_type = config.get("type")
        if block_type == "cnn":
            return CNNBlock(
                in_channels=config.get("in_channels"),
                out_channels=config.get("out_channels"),
                kernel_size=config.get("kernel_size"),
                stride=config.get("stride"),
                padding=config.get("padding"),
                pool=config.get("pool"),
                pool_kernel=config.get("pool_kernel"),
                is_1d=config.get("is_1d", False)
            )
        elif block_type == "resnet":
            return ResNetBlock(
                in_channels=config.get("in_channels"),
                out_channels=config.get("out_channels"),
                stride=config.get("stride", 1),
                is_1d=config.get("is_1d", False)
            )
        elif block_type == "gru":
            return GRUBlock(
                input_size=config.get("input_size"),
                hidden_size=config.get("hidden_size"),
                num_layers=config.get("num_layers"),
                dropout=config.get("dropout", 0.0)
            )
        elif block_type == "lstm":
            return LSTMBlock(
                input_size=config.get("input_size"),
                hidden_size=config.get("hidden_size"),
                num_layers=config.get("num_layers"),
                dropout=config.get("dropout", 0.0)
            )
        elif block_type == "fc":
            return FCBlock(
                in_features=config.get("in_features"),
                out_features=config.get("out_features"),
                dropout=config.get("dropout", 0.5)
            )
        elif block_type == "adaptive_pool":
            return AdaptivePoolBlock(
                output_size=config.get("output_size"),
                is_1d=config.get("is_1d", False)
            )
        elif block_type == "flatten":
            return FlattenBlock()
        else:
            raise ValueError(f"Unknown block type: {block_type}")


# --- Builder Pattern ---
class ModelBuilder:
    """Builder to construct a model from YAML configuration."""
    @staticmethod
    def build_model(config_path: str, num_classes: int) -> nn.Module:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get("model", {})
        
        input_type = config.get("input_type", "mel")
        blocks_config = config.get("blocks", [])
        
        layers = []
        for block_config in blocks_config:
            # Override out_features for the last FC block to match num_classes
            if block_config.get("type") == "fc" and block_config == blocks_config[-1]:
                block_config["out_features"] = num_classes
            layers.append(BlockFactory.create_block(block_config))
        
        return AudioClassifier(input_type, nn.Sequential(*layers))


# --- Main Model ---
class AudioClassifier(nn.Module):
    """Audio classification model built from modular blocks."""
    def __init__(self, input_type: str, blocks: nn.Sequential):
        super(AudioClassifier, self).__init__()
        self.input_type = input_type
        self.blocks = blocks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor ([batch, 16000] for raw, [batch, 8000] for fft, [batch, 128, time_frames] for mel).
        
        Returns:
            torch.Tensor: Class logits ([batch, num_classes]).
        """
        if self.input_type == "mel":
            x = x.unsqueeze(1)  # [batch, 128, time_frames] -> [batch, 1, 128, time_frames]
        elif self.input_type in ["raw", "fft"]:
            x = x.unsqueeze(1)  # [batch, seq_length] -> [batch, 1, seq_length]
        
        x = self.blocks(x)
        return x


if __name__ == "__main__":
    # Example usage
    model = ModelBuilder.build_model("configs/model_raw.yaml", num_classes=12)
    print(model)
    
    # Test with dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = torch.randn(32, 16000).to(device)  # Example for mel
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [32, 12]