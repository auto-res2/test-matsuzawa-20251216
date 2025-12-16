import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """Bottleneck residual block for ResNet-50"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.expansion = 4
        
        self.conv1 = nn.Conv2d(in_channels, out_channels // self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // self.expansion)
        
        self.conv2 = nn.Conv2d(
            out_channels // self.expansion, out_channels // self.expansion,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels // self.expansion)
        
        self.conv3 = nn.Conv2d(out_channels // self.expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 for arbitrary output dimensions"""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResidualBlock, 64, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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


class ResNet50(nn.Module):
    """ResNet-50 for arbitrary output dimensions"""
    
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BottleneckBlock, 64, 256, 3, stride=1)
        self.layer2 = self._make_layer(BottleneckBlock, 256, 512, 4, stride=2)
        self.layer3 = self._make_layer(BottleneckBlock, 512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, 1024, 2048, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        self._init_weights()
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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


class LSTMClassifier(nn.Module):
    """2-layer LSTM for sequence classification"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Reshape if needed (for flattened image inputs)
        if x.dim() == 2:
            batch_size = x.shape[0]
            seq_len = x.shape[1] // self.input_size
            if seq_len * self.input_size != x.shape[1]:
                x = x[:, :seq_len * self.input_size]
            x = x.view(batch_size, seq_len, self.input_size)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


def get_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """Factory function to create model based on config"""
    
    model_name = model_cfg.get("name", "ResNet-18").lower()
    num_classes = model_cfg.get("num_classes", 10)
    input_channels = model_cfg.get("input_channels", 3)
    
    if "resnet-18" in model_name or "resnet18" in model_name:
        return ResNet18(num_classes=num_classes, input_channels=input_channels)
    
    elif "resnet-50" in model_name or "resnet50" in model_name:
        return ResNet50(num_classes=num_classes, input_channels=input_channels)
    
    elif "lstm" in model_name:
        hidden_size = model_cfg.get("hidden_size", 256)
        num_layers = model_cfg.get("num_layers", 2)
        input_size = model_cfg.get("input_size", 100)
        dropout = model_cfg.get("dropout", 0.5)
        
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
