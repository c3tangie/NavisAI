# Neural Network Related Stuffs
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt

class NavisSteer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.dropout(X)

        X = self.pool(F.relu(self.conv2(X)))
        X = self.dropout(X)

        X = self.pool(F.relu(self.conv3(X)))
        X = self.dropout(X)

        X = self.global_pool(X)
        X = X.view(X.size(0), -1)

        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.fc2(X)

        return X
    
class NavisVision(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.dropout(X)

        X = self.pool(F.relu(self.conv2(X)))
        X = self.dropout(X)

        X = self.pool(F.relu(self.conv3(X)))
        X = self.dropout(X)

        X = self.global_pool(X)
        X = X.view(X.size(0), -1)

        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.fc2(X)

        return X
