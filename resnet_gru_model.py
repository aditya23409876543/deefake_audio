import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------
# ResNet + GRU + Attention Model (Open Source)
# ---------------------
class ResidualBlock(nn.Module):
    """Basic ResNet residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetGRUModel(nn.Module):
    """ResNet + GRU + Attention model for audio deepfake detection"""
    def __init__(self, num_classes=2):
        super().__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # GRU layers (bidirectional) - hidden size 256 gives 512 output
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, 
                         bidirectional=True, dropout=0.3)
        
        # Multi-head attention - embed_dim 512 matches GRU output
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)
        
        # Classifier - exact match to checkpoint structure
        # Indices with params: 0, 2, 4, 6, 8
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),           # 0
            nn.ReLU(inplace=True),         # 1 (no params)
            nn.BatchNorm1d(512),           # 2
            nn.ReLU(inplace=True),         # 3 (no params)  
            nn.Linear(512, 128),           # 4
            nn.ReLU(inplace=True),         # 5 (no params)
            nn.BatchNorm1d(128),           # 6
            nn.ReLU(inplace=True),         # 7 (no params)
            nn.Linear(128, num_classes)    # 8
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, spectral, mfcc, phase):
        # Use spectral input
        if spectral.dim() == 4:
            x = spectral[:, 0:1, :, :]  # Use first channel
        else:
            x = spectral.unsqueeze(1)
        
        # ResNet feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)  # 64 channels
        x = self.layer2(x)  # 128 channels
        x = self.layer3(x)  # 256 channels
        
        # Reshape for GRU: (batch, channels, height, width) -> (batch, height*width, channels)
        b, c, h, w = x.size()
        x = x.view(b, h * w, c)  # Flatten spatial, keep channels
        
        # GRU processing (input: batch, seq_len, features)
        x, _ = self.gru(x)  # x: (batch, seq_len, 512) since bidirectional
        
        # Attention
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, features)
        
        # Classifier
        x = self.classifier(x)
        return x

# ---------------------
# Model Factory
# ---------------------
def get_model(model_type='enhanced'):
    """
    Factory function to create models
    Supported types: 'enhanced', 'pytorch_model'
    """
    if model_type == 'enhanced':
        return DeepfakeDetectorCNN()
    elif model_type == 'pytorch_model':
        return ResNetGRUModel(num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
