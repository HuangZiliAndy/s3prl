"""
model.py — ASR acoustic model architectures shared by all ASR downstream tasks.

Provides two CTC-compatible encoder models:
  - TransformerASRModel: Transformer encoder with sinusoidal or learned positional encoding.
  - LSTMASRModel:        Bidirectional or unidirectional LSTM encoder.

Both follow the same interface:
    logits, lengths = model(features, lengths)
where features is (B, T, input_dim) and logits is (B, T, output_dim).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'.

    Adds fixed sine/cosine positional signals to the input embeddings.
    Even dimensions use sine, odd dimensions use cosine, with frequencies
    decreasing geometrically from 1 to 1/10000 across the model dimension.
    """

    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """Add positional encoding to x.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) with positional signals added.
        """
        return x + self.pe[:, :x.size(1)]


class TransformerASRModel(nn.Module):
    """Transformer encoder for CTC-based ASR.

    Projects input features to d_model, applies positional encoding, then
    runs a standard Transformer encoder with padding masking. A linear head
    projects encoder outputs to the output vocabulary size.

    Args:
        input_dim:       Dimension of input features (upstream model output).
        output_dim:      Output vocabulary size (number of CTC tokens).
        upstream_rate:   Frame shift of upstream model in samples (unused internally,
                         kept for API compatibility).
        d_model:         Transformer model dimension.
        nhead:           Number of attention heads.
        dim_feedforward: Feedforward sublayer dimension.
        dropout:         Dropout probability.
        activation:      Activation function for the feedforward sublayer ('relu' or 'gelu').
        num_layers:      Number of Transformer encoder layers.
        pos_enc:         Positional encoding type: 'sinusoidal' (fixed) or 'embedding' (learned).
        max_len:         Maximum sequence length for positional encoding.
    """

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
        """Run the Transformer encoder.

        Args:
            x:       (B, T, input_dim) padded feature tensor.
            lengths: (B,) IntTensor of valid frame counts per sequence.

        Returns:
            logits:  (B, T, output_dim) unnormalized CTC log-probabilities.
            lengths: (B,) unchanged input lengths.
        """
        B, T, _ = x.size()
        if self.pos_enc == "sinusoidal":
            x = self.pos_embedding(x)
        elif self.pos_enc == "embedding":
            x = x + self.pos_embedding[:, :T, :]
        else:
            raise ValueError("Positional encoding not defined.")

        # True where positions are padding; the Transformer will ignore these.
        mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1).to(x.device)

        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.output_proj(x)
        return x, lengths


class LSTMASRModel(nn.Module):
    """LSTM encoder for CTC-based ASR.

    Runs a (optionally bidirectional) multi-layer LSTM on packed sequences,
    then projects to the output vocabulary size.

    Args:
        input_dim:     Dimension of input features (upstream model output).
        output_dim:    Output vocabulary size (number of CTC tokens).
        upstream_rate: Frame shift of upstream model in samples (unused internally,
                       kept for API compatibility).
        d_model:       LSTM hidden size per direction.
        bidirectional: If True, use a bidirectional LSTM (output projected from d_model*2).
        dropout:       Dropout probability between LSTM layers (ignored if num_layers=1).
        num_layers:    Number of stacked LSTM layers.
    """

    def __init__(self, input_dim, output_dim, upstream_rate, d_model, bidirectional, dropout, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, d_model, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.output_proj = nn.Linear(d_model * 2, output_dim)
        else:
            self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x, lengths):
        """Run the LSTM encoder.

        Sequences must be sorted in descending length order (guaranteed by
        ASRDataset.collate_fn).

        Args:
            x:       (B, T, input_dim) padded feature tensor.
            lengths: (B,) IntTensor of valid frame counts per sequence.

        Returns:
            logits:  (B, T, output_dim) unnormalized CTC log-probabilities.
            lengths: (B,) unchanged input lengths.
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        packed_out, _ = self.lstm(packed)
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.output_proj(out_padded), lengths
