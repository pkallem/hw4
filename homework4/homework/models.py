from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class LinearPlanner(nn.Module):
    """
    A very simple "linear" planner with no hidden layers.
    Just flattens (track_left, track_right) -> outputs n_waypoints * 2
    """
    def __init__(self, n_track=10, n_waypoints=3):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Flatten (left/right) track: 2 sides * n_track points * 2 coords
        input_dim = 2 * n_track * 2
        output_dim = n_waypoints * 2
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        bsz = track_left.shape[0]
        # Concatenate left/right boundaries
        x = torch.cat([track_left, track_right], dim=1)  # (B, 2*n_track, 2)
        x = x.view(bsz, -1)                              # (B, 2*n_track*2)
        out = self.linear(x)                             # (B, n_waypoints*2)
        return out.view(bsz, self.n_waypoints, 2)        # (B, n_waypoints, 2)


class MLPPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3, hidden_dim=128):
        """
        A simple MLP with two hidden layers.
        """
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = 2 * n_track * 2  # left+right boundaries, each n_track points, 2 coords
        output_dim = n_waypoints * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        bsz = track_left.shape[0]
        x = torch.cat([track_left, track_right], dim=1)  # (B, 2*n_track, 2)
        x = x.view(bsz, -1)                              # (B, input_dim)
        out = self.net(x)                                # (B, n_waypoints*2)
        return out.view(bsz, self.n_waypoints, 2)        # (B, n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3, d_model=64, nhead=8, num_layers=2, dim_feedforward=256):
        """
        Example Transformer-based planner that uses cross attention.
        """
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Input embedding for track points
        self.input_embed = nn.Linear(2, d_model)

        # Waypoint query embeddings
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,  # Expect shape (seq, batch, embed)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer -> 2D coords
        self.output_fc = nn.Linear(d_model, 2)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        bsz = track_left.shape[0]

        # Concatenate left/right boundaries -> (B, 2*n_track, 2)
        track_points = torch.cat([track_left, track_right], dim=1)

        # Embed track points -> (B, 2*n_track, d_model)
        track_emb = self.input_embed(track_points)

        # The decoder expects (seq_len, batch, d_model)
        memory = track_emb.transpose(0, 1)  # (2*n_track, B, d_model)

        # Query embeddings: (n_waypoints, d_model) -> expand to (n_waypoints, B, d_model)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, bsz, 1)

        # Cross-attention: queries -> memory
        decoded = self.transformer_decoder(tgt=queries, memory=memory)  # (n_waypoints, B, d_model)
        out = self.output_fc(decoded)                                   # (n_waypoints, B, 2)
        out = out.transpose(0, 1)                                       # (B, n_waypoints, 2)
        return out


class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints=3):
        super().__init__()
        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN).float().view(1, -1, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD).float().view(1, -1, 1, 1), persistent=False)

        # A small CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After these layers, (3, 96, 128) -> (64, 12, 16)
        fc_in = 64 * 12 * 16

        self.fc = nn.Sequential(
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize
        x = (image - self.input_mean) / self.input_std
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x.view(x.shape[0], self.n_waypoints, 2)


##############################################
# Model factory + load/save utilities
##############################################

MODEL_FACTORY = {
    "linear_planner": LinearPlanner,
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def calculate_model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    """
    Called by the grader or user code to instantiate a model from MODEL_FACTORY
    and optionally load pretrained weights from <model_name>.th
    """
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model_name='{model_name}'. Must be one of {list(MODEL_FACTORY.keys())}")
    model_cls = MODEL_FACTORY[model_name]
    model = model_cls(**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find {model_path.name} in {HOMEWORK_DIR}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model_size_mb = calculate_model_size_mb(model)
    if model_size_mb > 20:
        raise AssertionError(f"Model '{model_name}' is too large: {model_size_mb:.2f} MB")

    return model


def save_model(model: nn.Module) -> str:
    """
    Use this function to save your model in the homework directory.
    The file name is automatically determined by the class type -> <model_name>.th
    """
    model_name = None
    for name, cls in MODEL_FACTORY.items():
        if isinstance(model, cls):
            model_name = name
            break
    if model_name is None:
        raise ValueError(f"Model type '{type(model)}' not recognized in MODEL_FACTORY")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return str(output_path)
