import torch
import torch.nn as nn
import torchaudio


class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == "conv1d":
            return Conv1D(**kwargs)
        elif model_type == "conv_rnn":
            return ConvRNN(**kwargs)
        elif model_type == "lstm":
            return LSTM(**kwargs)
        elif model_type == "transformer":
            return Transformer(**kwargs)
        elif model_type == "resnet":
            return ResNet(**kwargs)
        elif model_type == "resnet_rnn":
            return ResNetRNN(**kwargs)
        elif model_type == "sincnet":
            return SincNet(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class Conv1D(nn.Module):
    def __init__(self, n_classes, sample_rate, in_channels=1):
        super(Conv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)
        
        expected_length = sample_rate
        length = expected_length // 4 // 4
        length = (length - 2) // 1
        length = (length - 2) // 1
        if length <= 0:
            raise ValueError(f"Input length too short for sample_rate={sample_rate}")
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class ConvRNN(nn.Module):
    def __init__(self, n_classes, sample_rate, in_channels=1, hidden_size=128, num_layers=2):
        super(ConvRNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Calculate the output length after convolutions
        expected_length = sample_rate  # e.g., 16000 for 1-second clip
        length = expected_length // 4 // 4  # After conv1 and pool
        length = (length - 2) // 1  # After conv2
        length = (length - 2) // 1  # After conv3
        if length <= 0:
            raise ValueError(f"Input length too short for sample_rate={sample_rate}")
        
        self.sequence_length = length
        self.feature_size = 64  # Number of channels after conv3
        
        # Linear layer to project conv output to hidden_size
        self.projection = nn.Linear(self.feature_size, hidden_size)
        
        # GRU expects input of shape [batch_size, sequence_length, hidden_size]
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        # x: [batch_size, in_channels=1, sample_rate * duration]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x: [batch_size, 64, sequence_length]
        
        # Transpose to [batch_size, sequence_length, feature_size]
        x = x.transpose(1, 2)
        # x: [batch_size, sequence_length, 64]
        
        # Project to hidden_size
        x = self.projection(x)
        # x: [batch_size, sequence_length, hidden_size]
        
        # GRU
        output, hn = self.rnn(x)
        # output: [batch_size, sequence_length, hidden_size]
        
        # Take the last time step
        x = output[:, -1, :]
        x = self.fc(x)
        return x


class LSTM(nn.Module):
    def __init__(self, n_classes, sample_rate, in_channels=1, hidden_size=128, num_layers=2):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=sample_rate, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        x = x.squeeze(1)
        output, (hn, cn) = self.rnn(x)
        x = output[:, -1, :]
        x = self.fc(x)
        return x


class Transformer(nn.Module):
    def __init__(self, n_classes, sample_rate, in_channels=1, d_model=64, nhead=4, num_layers=2):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(sample_rate, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_classes)
    
    def forward(self, x):
        x = x.squeeze(1)
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    def __init__(self, n_classes, sample_rate, in_channels=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.res_block1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64)
        )
        self.res_block2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)
        
        expected_length = sample_rate
        length = expected_length // 4 // 4
        if length <= 0:
            raise ValueError(f"Input length too short for sample_rate={sample_rate}")
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        res = x
        x = self.res_block1(x)
        x = x + res
        x = self.relu(x)
        res = x
        x = self.res_block2(x)
        x = x + res
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class ResNetRNN(nn.Module):
    def __init__(self, n_classes, sample_rate, in_channels=1, hidden_size=128, num_layers=2):
        super(ResNetRNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.res_block1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64)
        )
        self.res_block2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64)
        )
        
        expected_length = sample_rate
        length = expected_length // 4 // 4
        if length <= 0:
            raise ValueError(f"Input length too short for sample_rate={sample_rate}")
        
        self.sequence_length = length
        self.feature_size = 64
        
        self.projection = nn.Linear(self.feature_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        res = x
        x = self.res_block1(x)
        x = x + res
        x = self.relu(x)
        res = x
        x = self.res_block2(x)
        x = x + res
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        output, (hn, cn) = self.rnn(x)
        x = output[:, -1, :]
        x = self.fc(x)
        return x


class SincNet(nn.Module):
    def __init__(self, n_classes, sample_rate, in_channels=1):
        super(SincNet, self).__init__()
        self.sinc_conv = nn.Conv1d(in_channels, 80, kernel_size=251, stride=1, padding=125)
        self.bn1 = nn.BatchNorm1d(80)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(80, 60, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(60)
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(60)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(60, n_classes)
        
        expected_length = sample_rate
        length = (expected_length - 2) // 3
        length = (length - 4) // 1
        length = (length - 4) // 1
        if length <= 0:
            raise ValueError(f"Input length too short for sample_rate={sample_rate}")
    
    def forward(self, x):
        x = self.sinc_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x