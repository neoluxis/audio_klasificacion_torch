import torch
import torch.nn as nn


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


class Transformer(nn.Module):
    """Transformer for audio classification."""
    def __init__(self, n_classes, sample_rate=16000, in_channels=1, d_model=64, nhead=4, num_layers=2):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(in_channels * sample_rate, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        out = self.fc(x)
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
        else:
            raise ValueError(f"Unknown model type: {model_type}")