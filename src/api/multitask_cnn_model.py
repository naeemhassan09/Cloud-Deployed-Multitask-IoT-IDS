import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone1D(nn.Module):
    def __init__(self, num_features, conv_channels=64, conv_channels_mid=128, rep_dim=128, dropout=0.3):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(conv_channels, conv_channels_mid, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels_mid),
            nn.ReLU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(conv_channels_mid, conv_channels_mid, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels_mid),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj = nn.Sequential(
            nn.Linear(conv_channels_mid, rep_dim),
            nn.BatchNorm1d(rep_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.rep_dim = rep_dim

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x = x.mean(dim=2)
        x = self.proj(x)
        return x


class MultiTaskCNN1D(nn.Module):
    def __init__(self, num_features, num_attacks, num_devices, rep_dim=128):
        super().__init__()
        self.backbone = CNNBackbone1D(num_features, rep_dim=rep_dim)
        self.attack_head = nn.Linear(rep_dim, num_attacks)
        self.device_head = nn.Linear(rep_dim, num_devices)

    def forward(self, x):
        rep = self.backbone(x)
        logits_att = self.attack_head(rep)
        logits_dev = self.device_head(rep)
        return logits_att, logits_dev