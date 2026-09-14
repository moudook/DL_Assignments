import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Explicit weight initialisation
# ---------------------------------------------------------------------------
# Xavier Glorot Uniform: limit = sqrt(6 / (fan_in + fan_out)).
# Textbook choice for tanh, matches TF/Keras default, and is applied
# explicitly so we own the formula and range.
#
# Seed policy: torch.manual_seed(INIT_SEED) is called ONCE in get_models()
# so every architecture draws from one reproducible global stream.  The
# state_dict save/load in experiment.py guarantees that every optimizer
# sees the exact same starting weights for a given architecture.
# ---------------------------------------------------------------------------

INIT_SEED = 42


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class FCNN(nn.Module):
    def __init__(self, input_dim=784, hidden_sizes=[256, 128, 64], num_classes=5):
        """
        Generic FCNN with a configurable number of hidden layers and nodes per layer.

        Args:
            input_dim (int): Input dimensionality (784 for flattened 28x28 MNIST images).
            hidden_sizes (list[int]): Number of nodes in each hidden layer.
                                      len(hidden_sizes) == number of hidden layers.
            num_classes (int): Number of output classes (5 for this assignment).
        """
        super(FCNN, self).__init__()
        layers = []
        current_in = input_dim

        # Build hidden layers dynamically based on hidden_sizes
        for h in hidden_sizes:
            layers.append(nn.Linear(current_in, h))
            layers.append(nn.Tanh())   # Tanh activation for hidden layers
            current_in = h

        # Output layer — raw logits; CrossEntropyLoss applies log-softmax internally
        layers.append(nn.Linear(current_in, num_classes))

        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x)


def _make_model(hidden_sizes, input_dim=784, num_classes=5):
    """
    Create an FCNN.  The RNG is NOT reset here; get_models() sets the seed
    once before building all architectures so each model consumes a unique
    deterministic slice of the global random stream.
    """
    return FCNN(input_dim=input_dim, hidden_sizes=hidden_sizes, num_classes=num_classes)


def get_models():
    """
    Returns FCNN architectures covering 3 depths (3/4/5 hidden layers) ×
    3 patterns (Funnel/Uniform/Bottleneck) × 3 node-size scales (Large/Medium/Small).
    = 27 architectures total.

    Patterns:
      Funnel     — neurons strictly decrease layer-by-layer (pyramidal)
      Uniform    — all hidden layers have the same width
      Bottleneck — neurons decrease to a narrow middle then expand back

    Scales (approx):
      Large  — first hidden layer ~512 nodes
      Medium — first hidden layer ~256 nodes
      Small  — first hidden layer ~128 nodes

    All models are created from a single global random stream seeded once
    with INIT_SEED, so weight initialisation is reproducible and explicit.
    """
    torch.manual_seed(INIT_SEED)
    print(f"[models] Initializing 27 architectures with Xavier Glorot Uniform (seed={INIT_SEED})")
    return {
        # ════════════════════════════════════════════════════════════════════
        # 3 Hidden Layers
        # ════════════════════════════════════════════════════════════════════

        # — Funnel (decreasing) —
        '3L_Funnel_Large':       _make_model([512, 256, 128]),
        '3L_Funnel_Medium':      _make_model([256, 128,  64]),
        '3L_Funnel_Small':       _make_model([128,  64,  32]),

        # — Uniform (same width) —
        '3L_Uniform_Large':      _make_model([512, 512, 512]),
        '3L_Uniform_Medium':     _make_model([256, 256, 256]),
        '3L_Uniform_Small':      _make_model([128, 128, 128]),

        # — Bottleneck (wide → narrow → wide) —
        '3L_Bottleneck_Large':   _make_model([512,  64, 512]),
        '3L_Bottleneck_Medium':  _make_model([256,  64, 256]),
        '3L_Bottleneck_Small':   _make_model([128,  32, 128]),

        # ════════════════════════════════════════════════════════════════════
        # 4 Hidden Layers
        # ════════════════════════════════════════════════════════════════════

        # — Funnel —
        '4L_Funnel_Large':       _make_model([512, 256, 128,  64]),
        '4L_Funnel_Medium':      _make_model([256, 128,  64,  32]),
        '4L_Funnel_Small':       _make_model([128,  64,  32,  16]),

        # — Uniform —
        '4L_Uniform_Large':      _make_model([512, 512, 512, 512]),
        '4L_Uniform_Medium':     _make_model([256, 256, 256, 256]),
        '4L_Uniform_Small':      _make_model([128, 128, 128, 128]),

        # — Bottleneck (wide → squeeze → expand) —
        '4L_Bottleneck_Large':   _make_model([512, 128, 128, 512]),
        '4L_Bottleneck_Medium':  _make_model([256,  64,  64, 256]),
        '4L_Bottleneck_Small':   _make_model([128,  32,  32, 128]),

        # ════════════════════════════════════════════════════════════════════
        # 5 Hidden Layers
        # ════════════════════════════════════════════════════════════════════

        # — Funnel —
        '5L_Funnel_Large':       _make_model([512, 256, 128,  64,  32]),
        '5L_Funnel_Medium':      _make_model([256, 128,  64,  32,  16]),
        '5L_Funnel_Small':       _make_model([128,  64,  32,  16,   8]),

        # — Uniform —
        '5L_Uniform_Large':      _make_model([512, 512, 512, 512, 512]),
        '5L_Uniform_Medium':     _make_model([256, 256, 256, 256, 256]),
        '5L_Uniform_Small':      _make_model([128, 128, 128, 128, 128]),

        # — Bottleneck (wide → squeeze in middle → expand) —
        '5L_Bottleneck_Large':   _make_model([512, 256,  64, 256, 512]),
        '5L_Bottleneck_Medium':  _make_model([256, 128,  32, 128, 256]),
        '5L_Bottleneck_Small':   _make_model([128,  64,  16,  64, 128]),
    }
