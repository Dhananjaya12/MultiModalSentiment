import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Adds position information to each timestep.
    Without this, the transformer can't tell row 1 from row 499.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time_steps, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ModalityEncoder(nn.Module):
    """
    Encodes one modality's raw features over time.

    What it does:
      1. Projects raw features to d_model  e.g. 74 → 128
      2. Adds positional encoding
      3. Runs a Transformer to build temporal context

    Audio:  (batch, 500, 74)  → (batch, 500, 128)
    Vision: (batch, 500, 713) → (batch, 500, 128)
    Text:   (batch, 128, 768) → (batch, 128, 128)
    """
    def __init__(self, input_dim: int, cfg):
        super().__init__()

        self.cfg = cfg
        d_model    = self.cfg['d_model']
        nhead      = self.cfg['n_heads']
        num_layers = self.cfg['enc_layers']
        dropout    = self.cfg['dropout']

        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= d_model * 4,
            dropout        = dropout,
            batch_first    = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        # x: (batch, T, input_dim)
        x = self.projection(x)        # → (batch, T, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x                      # (batch, T, d_model)