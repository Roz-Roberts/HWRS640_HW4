import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 1,
            output_size: int = 1,
            dropout: float = 0.0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Input Format: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0.0)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        last_timestep = lstm_out[:, -1, :]  # Batch, hidden_size

        y_pred = self.fc(last_timestep)

        if self.output_size == 1:
            y_pred = y_pred.squeeze(-1)

        return y_pred


### MODEL FORMAT BELLOW TO CALL ELSEWHERE
# model = LSTM_model(input_size=5,      # prcp, tmax, tmin, srad, vp
#     hidden_size=64,
#     num_layers=1,
#     output_size=1)

