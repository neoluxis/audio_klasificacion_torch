import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SincConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1):
        super(SincConv1d, self).__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports one input channel")
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize filterbanks
        low_freq = 50
        high_freq = sample_rate / 2 - (low_freq + 1)
        mel = np.linspace(self.to_mel(low_freq), self.to_mel(high_freq), out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n = torch.linspace(0, kernel_size - 1, steps=kernel_size) - (kernel_size - 1) / 2
        self.register_buffer("n", n)

        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def to_hz(self, mel):
        return 700 * (10**(mel / 2595) - 1)

    def sinc(self, x):
        return torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(x) / x)

    def forward(self, x):
        low = torch.clamp(torch.abs(self.low_hz), min=1.0, max=self.sample_rate / 2)
        band = torch.clamp(torch.abs(self.band_hz), min=1e-3)  # Prevent near-zero band
        high = torch.clamp(low + band, min=1.0, max=self.sample_rate / 2)
        band = (high - low).view(-1, 1)

        f_times_t_low = 2 * np.pi * low * self.n / self.sample_rate
        f_times_t_high = 2 * np.pi * high * self.n / self.sample_rate

        band_pass = self.sinc(f_times_t_high) - self.sinc(f_times_t_low)
        band_pass = band_pass * self.window
        band_pass = band_pass / torch.clamp(2 * band, min=1e-6)  # Safer normalization

        # Normalize filters to prevent large outputs
        band_pass = band_pass / (torch.norm(band_pass, dim=-1, keepdim=True) + 1e-8)

        # Debug NaNs
        if torch.isnan(band_pass).any():
            print("NaN detected in band_pass:", band_pass)
        if torch.isinf(band_pass).any():
            print("Inf detected in band_pass:", band_pass)

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        out = F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)

        if torch.isnan(out).any():
            print("NaN detected in conv output:", out)

        return out
    
class SincNet(nn.Module):
    def __init__(self, sample_rate=16000, n_classes=10, in_channels=1):
        super(SincNet, self).__init__()
        self.features = nn.Sequential(
            SincConv1d(out_channels=80, kernel_size=251, sample_rate=sample_rate, in_channels=in_channels),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(80, 60, kernel_size=5),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(60, 60, kernel_size=5),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(60, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class BasicBlock(nn.Module):
    """Basic residual block for 1D ResNet."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
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


class ResNet(nn.Module):
    """1D ResNet adapted for audio classification."""
    def __init__(self, block, layers, n_classes, sample_rate=16000, in_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNetRNN(nn.Module):
    """Hybrid ResNet + LSTM for audio classification."""
    def __init__(self, block, layers, n_classes, sample_rate=16000, in_channels=1, hidden_size=128, num_layers=2):
        super(ResNetRNN, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # Calculate feature dimension after ResNet
        self.feature_dim = 256 * block.expansion
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Reshape for LSTM: [batch, channels, time] -> [batch, time, channels]
        x = x.permute(0, 2, 1)
        # LSTM expects [batch, seq_len, input_size]
        output, (hn, cn) = self.lstm(x)
        # Use last hidden state
        out = self.fc(output[:, -1, :])
        return out


class Conv1d(nn.Module):
    """Basic 1D CNN for audio classification."""
    def __init__(self, n_classes, sample_rate=16000, in_channels=1):
        super(Conv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # Calculate output size: 16000 / (2^3) * 64 = 2000 * 64
        self.fc1 = nn.Linear(64 * (sample_rate // 8), 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ConvRNN(nn.Module):
    """CNN + RNN for audio classification."""
    def __init__(self, n_classes, sample_rate=16000, in_channels=1, hidden_size=128, num_layers=2):
        super(ConvRNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(input_size=32 * (sample_rate // 4), hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        output, (hn, cn) = self.rnn(x)
        out = self.fc(output[:, -1, :])
        return out


class LSTM(nn.Module):
    """LSTM for audio classification."""
    def __init__(self, n_classes, sample_rate=16000, in_channels=1, hidden_size=128, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels * sample_rate, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        output, (hn, cn) = self.lstm(x)
        out = self.fc(output[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class Transformer(nn.Module):
    """Improved Transformer for audio classification."""
    def __init__(self, n_classes, sample_rate=16000, in_channels=1, d_model=128, nhead=4, num_layers=4, conv_channels=64):
        super(Transformer, self).__init__()

        # Encode raw waveform using Conv1d
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_channels, d_model, kernel_size=5, stride=4, padding=2),  # output: [B, d_model, T/16]
            nn.ReLU()
        )

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: [B, C=1, T=16000]
        x = self.encoder(x)  # → [B, d_model, T']
        x = x.permute(0, 2, 1)  # → [B, T', d_model]
        x = self.pos_encoder(x)
        x = self.transformer(x)  # → [B, T', d_model]
        x = x.mean(dim=1)  # Global Average Pooling
        out = self.classifier(x)
        return out


class ModelFactory:
    """Factory to create model instances."""
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == "conv1d":
            return Conv1d(**kwargs)
        elif model_type == "conv_rnn":
            return ConvRNN(**kwargs)
        elif model_type == "lstm":
            return LSTM(**kwargs)
        elif model_type == "transformer":
            return Transformer(**kwargs)
        elif model_type == "resnet":
            return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)  # ResNet18-like
        elif model_type == "resnet_rnn":
            return ResNetRNN(BasicBlock, [2, 2, 2], **kwargs)  # Shallower ResNet + LSTM
        elif model_type == "sincnet":
            return SincNet(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        
if __name__ == "__main__":
    # Example usage
    model = ModelFactory.create_model("sincnet", n_classes=10, sample_rate=16000, in_channels=1)
    print(model)
    x = torch.randn(8, 1, 16000)  # Batch of 8 audio samples
    output = model(x)
    print(output.shape)  # Should be [8, 10] for 10 classes