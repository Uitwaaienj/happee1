import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import os

class BaseModel(nn.Module, ABC):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class RNNModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super().__init__(input_size, hidden_size, num_layers, num_classes, dropout)
        self.bn = nn.BatchNorm1d(input_size)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对每个时间步进行批归一化
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = self.bn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class GRUModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super().__init__(input_size, hidden_size, num_layers, num_classes, dropout)
        self.bn = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class LSTMModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super().__init__(input_size, hidden_size, num_layers, num_classes, dropout)
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class CNNLSTMModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super().__init__(input_size, hidden_size, num_layers, num_classes, dropout)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.bn2(x)
        
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.dropout2(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class CNNGRUModel(BaseModel):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super().__init__(input_size, hidden_size, num_layers, num_classes, dropout)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(128)
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.bn2(x)
        
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = self.dropout2(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class ModelFactory:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

    def create_model(self, model_type: str) -> nn.Module:
        if model_type == 'cnn_gru':
            return CNNGRU(self.input_size, self.hidden_size, self.num_layers, self.num_classes, self.dropout)
        elif model_type == 'cnn_lstm':
            return CNNLSTM(self.input_size, self.hidden_size, self.num_layers, self.num_classes, self.dropout)
        elif model_type == 'cnn':
            return CNN(self.input_size, self.hidden_size, self.num_layers, self.num_classes, self.dropout)
        elif model_type == 'gru':
            return GRU(self.input_size, self.hidden_size, self.num_layers, self.num_classes, self.dropout)
        elif model_type == 'lstm':
            return LSTM(self.input_size, self.hidden_size, self.num_layers, self.num_classes, self.dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class CNNGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super(CNNGRU, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.gru = nn.GRU(128, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
        x, _ = self.gru(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

class CNNLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(3, 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(2 * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x 