import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN_LSTM(nn.Module):
    def __init__(self, num_channels, num_classes, window_length):
        super(CNN_LSTM, self).__init__()

        #ARchitecture
        
        #https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, padding= 3 // 2)

        self.pool = nn.MaxPool1d(kernel_size=2)  # halves the sequence length
        #self.dropout = nn.Dropout(0.5)# for regularisation, was uded in famust ALEXNET

        # The LSTM will receive a sequence where each time step is represented by a 64-dimensional feature vector.
        #wonder if pooling kills the actual timestep data....
        lstm_input_size = 64
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    
    def forward(self,x):
        # x: (batch_size, window_length, num_channels)
        # Permute to (batch_size, num_channels, window_length) for Conv1d
         # Permute for Conv1D: (B, C, T)
        x = x.permute(0, 2, 1)
        #mnum channels is 6 for us
        x = self.conv1(x) # (Batch, 64, window_length)
        x = F.relu(x)
        x = self.pool(x) #(Batch, 64, window_length//2)
        #x = self.dropout(x)

        # Prepare for LSTM: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)  # now (batch_size, window_length/2, 64)
        # LSTM

        lstm_out, (hn, cn) = self.lstm(x)
        final_hidden = hn[-1]  # Shape: (batch_size, 128)
        out = self.fc1(final_hidden)  # (batch_size, 64)
        out = F.relu(out)
        out = self.fc2(out) # (batch_size, num_classes)
        return out