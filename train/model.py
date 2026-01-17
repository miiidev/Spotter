import torch
import torch.nn as nn


class DeepfakeLSTM(nn.Module):
    """
    BiLSTM-based deepfake detector.
    Input shape:  (B, T, D)
    Output shape: (B, 1)
    """

    def __init__(
        self,
        input_dim=2808,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        x: [B, T, D]
        """
        _, (h_n, _) = self.lstm(x)

        # Last layer, both directions
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat((h_forward, h_backward), dim=1)

        logits = self.classifier(h)
        return logits.squeeze(1)
