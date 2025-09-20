import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerASRModel(nn.Module):
    def __init__(self, input_dim, output_dim, upstream_rate, d_model, nhead, dim_feedforward, dropout, activation, num_layers, pos_enc="sinusoidal", max_len=2048):
        super(TransformerASRModel, self).__init__()

        self.pos_enc = pos_enc
        if self.pos_enc == "sinusoidal":
            self.pos_embedding = PositionalEncoding(d_model, max_len)
        elif self.pos_enc == "embedding":
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        else:
            raise ValueError("Positional encoding not defined.")

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x, lengths):
        B, T, _ = x.size()
        if self.pos_enc == "sinusoidal":
            x = self.pos_embedding(x)
        elif self.pos_enc == "embedding":
            x = x + self.pos_embedding[:, :T, :]
        else:
            raise ValueError("Positional encoding not defined.")

        mask = torch.arange(T, device=x.device).expand(B, T) >= (lengths.unsqueeze(1)).to(x.device)

        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.output_proj(x)
        return x, lengths


class LSTMASRModel(nn.Module):
    def __init__(self, input_dim, output_dim, upstream_rate, d_model, bidirectional, dropout, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, d_model, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.output_proj = nn.Linear(d_model * 2, output_dim)
        else:
            self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # RNN
        packed_out, _ = self.lstm(packed)

        # Unpack back to padded sequence
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True)

        x = self.output_proj(out_padded)
        return x, lengths
