"""RNN / LSTM sequence classifier (many-to-one).

Follows the lecture slides:
  Hidden layers at each time step (RNN slides 19-20: the same network is applied at every time step,
  and s_t is its last hidden layer):  a_t = tanh(D x_t + e), applied to every x_t with shared D, e
  RNN layer:  s_t = tanh(U a_t + W s_{t-1} + b),  s_0 = 0
  LSTM layer: gates from a_t and h_{t-1}; s_t = f_t * s_{t-1} + i_t * s~_t;  h_t = o_t * tanh(s_t)
  "Two or more layer RNN - Task: Classification": the states of layer 1 are the inputs of layer 2,
  and the class is predicted once, from the last time step of the top layer:
  y = softmax(V s_T + c).
With no per-time-step hidden layers (dense_sizes = []), a_t is simply x_t.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SeqClassifier(nn.Module):
    def __init__(self, cell, input_dim, hidden_sizes, n_classes, dense_sizes=()):
        """hidden_sizes: neurons in each recurrent (RNN/LSTM) layer.
        dense_sizes: neurons in each hidden layer applied at every time step before the recurrent layers."""
        super().__init__()
        self.cell = cell
        layer_input = input_dim

        # Hidden layers of the network at a time step (same weights at every time step).
        self.dense_layers = nn.ModuleList()
        for size in dense_sizes:
            self.dense_layers.append(nn.Linear(layer_input, size))  # D and e
            layer_input = size

        # One single-layer RNN/LSTM per entry of hidden_sizes, each with its own U, W, b.
        self.layers = nn.ModuleList()
        for hidden in hidden_sizes:
            if cell == "rnn":
                layer = nn.RNN(layer_input, hidden, nonlinearity="tanh", batch_first=True)
            elif cell == "lstm":
                layer = nn.LSTM(layer_input, hidden, batch_first=True)
            else:
                raise ValueError(f"Unknown cell type {cell}")
            self.layers.append(layer)
            layer_input = hidden
        self.output = nn.Linear(layer_input, n_classes)  # V and c

    def forward(self, x, lengths):
        """x: (batch, max_len, input_dim) padded; lengths: true length of each sequence.

        Packing makes every recurrent layer stop at each sequence's own last time step,
        so padding never enters the state. Returns logits (softmax is inside the loss).
        """
        for dense in self.dense_layers:
            x = torch.tanh(dense(x))  # nn.Linear acts on the last dimension: every time step separately
        sequence = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        for layer in self.layers:
            sequence, last_state = layer(sequence)
        if self.cell == "lstm":
            last_state = last_state[0]  # LSTM returns (h_T, s_T); the output is h_T
        final_state = last_state[0]     # (1, batch, hidden) -> (batch, hidden)
        return self.output(final_state)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
