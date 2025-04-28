import torch
import torch.nn as nn
import math


class Conv1DModel(nn.Module):
    def __init__(self, n_classes, sample_rate=16000, in_channels=1):
        """
        A simple Conv1D model for audio classification.

        Args:
            n_classes (int): Number of output classes.
            sample_rate (int): Input sample rate (default: 16000 Hz).
            in_channels (int): Number of input channels (default: 1 for mono).
        """
        super(Conv1DModel, self).__init__()
        self.sample_rate = sample_rate
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        
        self.fc = nn.Linear(64 * (16000 // 8), n_classes)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvRNNModel(nn.Module):
    def __init__(self, n_classes, sample_rate=16000, in_channels=1, hidden_size=128, num_layers=2):
        """
        A Conv1D + GRU model for audio classification.

        Args:
            n_classes (int): Number of output classes.
            sample_rate (int): Input sample rate (default: 16000 Hz).
            in_channels (int): Number of input channels (default: 1 for mono).
            hidden_size (int): GRU hidden size (default: 128).
            num_layers (int): Number of GRU layers (default: 2).
        """
        super(ConvRNNModel, self).__init__()
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),
        )
        
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class LSTMModel(nn.Module):
    def __init__(self, n_classes, sample_rate=16000, in_channels=1, hidden_size=128, num_layers=2):
        """
        A Conv1D + LSTM model for audio classification.

        Args:
            n_classes (int): Number of output classes.
            sample_rate (int): Input sample rate (default: 16000 Hz).
            in_channels (int): Number of input channels (default: 1 for mono).
            hidden_size (int): LSTM hidden size (default: 128).
            num_layers (int): Number of LSTM layers (default: 2).
        """
        super(LSTMModel, self).__init__()
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Conv1D to reduce sequence length
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Fully connected: 2 * hidden_size due to bidirectional
        self.fc = nn.Linear(2 * hidden_size, n_classes)
        
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 16000].

        Returns:
            torch.Tensor: Logits of shape [batch_size, n_classes].
        """
        # Conv: [batch, 1, 16000] -> [batch, 32, 1000]
        x = self.conv_layers(x)
        
        # Reshape: [batch, 1000, 32]
        x = x.permute(0, 2, 1)
        
        # LSTM: [batch, 1000, 32] -> [batch, 1000, 2*hidden_size]
        x, _ = self.lstm(x)
        
        # Last time step: [batch, 2*hidden_size]
        x = x[:, -1, :]
        
        # FC: [batch, n_classes]
        x = self.fc(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, n_classes, sample_rate=16000, in_channels=1, d_model=128, nhead=4, num_layers=2):
        """
        A Transformer encoder model for audio classification.

        Args:
            n_classes (int): Number of output classes.
            sample_rate (int): Input sample rate (default: 16000 Hz).
            in_channels (int): Number of input channels (default: 1 for mono).
            d_model (int): Transformer embedding dimension (default: 128).
            nhead (int): Number of attention heads (default: 4).
            num_layers (int): Number of transformer layers (default: 2).
        """
        super(TransformerModel, self).__init__()
        self.sample_rate = sample_rate
        self.d_model = d_model
        
        # Conv1D to reduce sequence length and project to d_model
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8),  # 16000 / 8 = 2000
            nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 2000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.fc = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 16000].

        Returns:
            torch.Tensor: Logits of shape [batch_size, n_classes].
        """
        # Conv: [batch, 1, 16000] -> [batch, d_model, 2000]
        x = self.input_projection(x)
        
        # Reshape: [batch, 2000, d_model]
        x = x.permute(0, 2, 1)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Transformer: [batch, 2000, d_model] -> [batch, 2000, d_model]
        x = self.transformer_encoder(x)
        
        # Mean pool: [batch, d_model]
        x = x.mean(dim=1)
        
        # FC: [batch, n_classes]
        x = self.fc(x)
        return x


class ModelFactory:
    @staticmethod
    def create_model(model_name, **kwargs):
        """
        Factory method to create a model instance.

        Args:
            model_name (str): Name of the model ('conv1d', 'conv_rnn', 'lstm', 'transformer').
            **kwargs: Model-specific parameters (e.g., n_classes, sample_rate).

        Returns:
            nn.Module: Instantiated model.

        Raises:
            ValueError: If model_name is invalid.
        """
        model_name = model_name.lower()
        models = {
            "conv1d": Conv1DModel,
            "conv_rnn": ConvRNNModel,
            "lstm": LSTMModel,
            "transformer": TransformerModel,
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
        
        return models[model_name](**kwargs)


if __name__ == "__main__":
    # Example usage
    n_classes = 2
    sample_rate = 16000
    
    # Test all models
    model_configs = [
        ("conv1d", {"n_classes": n_classes, "sample_rate": sample_rate}),
        ("conv_rnn", {"n_classes": n_classes, "sample_rate": sample_rate, "hidden_size": 128, "num_layers": 2}),
        ("lstm", {"n_classes": n_classes, "sample_rate": sample_rate, "hidden_size": 128, "num_layers": 2}),
        ("transformer", {"n_classes": n_classes, "sample_rate": sample_rate, "d_model": 128, "nhead": 4, "num_layers": 2}),
    ]
    
    dummy_input = torch.randn(4, 1, 16000)
    
    for model_name, kwargs in model_configs:
        model = ModelFactory.create_model(model_name, **kwargs)
        print(f"{model_name.upper()} Model:", model)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"{model_name.upper()} output shape: {output.shape}")