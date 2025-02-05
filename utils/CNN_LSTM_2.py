import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_2(nn.Module):
    def __init__(self, num_channels, num_classes, window_length):
        super(CNN_LSTM_2, self).__init__()
        
        # 3 x Conv1D layers (no pooling)
        # Example configuration: all produce out_channels=64
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # 2-layer LSTM
        # The input to the LSTM will be 64 features (from the 3rd conv layer).
        # hidden_size=128 is an example; you can tweak it as needed.
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, window_length, num_channels)
        Example: (B, T, C) = (B, 30, 6)
        """
        # Permute for Conv1D: (batch_size, channels, time)
        x = x.permute(0, 2, 1)  # => (B, C, T)

        # 3 x Conv1D layers with ReLU, no pooling
        x = self.conv1(x)       # => (B, 64, T)
        x = F.relu(x)
        
        x = self.conv2(x)       # => (B, 64, T)
        x = F.relu(x)
        
        x = self.conv3(x)       # => (B, 64, T)
        x = F.relu(x)
        
        # Prepare for LSTM: (batch_size, time, features)
        # Now channels=64 is "features," time dimension is T
        x = x.permute(0, 2, 1)  # => (B, T, 64)

        # 2-layer LSTM
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (B, T, 128) if batch_first=True

        # Use the final hidden state of the top LSTM layer:
        # hn shape: (num_layers, B, hidden_size) -> (2, B, 128)
        final_hidden = hn[-1]  # => (B, 128)

        # Fully connected layers
        out = self.fc1(final_hidden)    # => (B, 64)
        out = F.relu(out)
        out = self.fc2(out)             # => (B, num_classes)

        return out