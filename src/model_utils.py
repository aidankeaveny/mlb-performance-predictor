import torch
import torch.nn as nn

class PlayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0, activation='relu', use_layernorm=False, use_residuals=False):
        super().__init__()
        self.use_residuals = use_residuals
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = []

        last_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(nn.Linear(last_dim, dim))
            self.norms.append(nn.LayerNorm(dim) if use_layernorm else None)
            self.activations.append(nn.GELU() if activation == 'gelu' else nn.ReLU())
            last_dim = dim

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(last_dim, 1)

    def forward(self, x):
        for layer, norm, act in zip(self.layers, self.norms, self.activations):
            residual = x
            x = layer(x)
            if norm:
                x = norm(x)
            x = act(x)
            x = self.dropout(x)
            if self.use_residuals and residual.shape == x.shape:
                x = x + residual
        return self.out(x)
