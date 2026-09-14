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
    Returns 5 strictly decreasing (funnel) FCNN architectures:
      - 2 × 3-hidden-layer
      - 2 × 4-hidden-layer
      - 1 × 5-hidden-layer

    All architectures end with 16 neurons in the final hidden layer
    (max 32 in later layers), sized for the dataset:
    11,385 train / 3,795 val / 3,795 test, 5 classes, input_dim=784.

    All models are created from a single global random stream seeded once
    with INIT_SEED, so weight initialisation is reproducible and explicit.
    """
    torch.manual_seed(INIT_SEED)
    print(f"[models] Initializing 5 architectures with Xavier Glorot Uniform (seed={INIT_SEED})")
    return {
        # ════════════════════════════════════════════════════════════════════
        # 3 Hidden Layers  (2 architectures)
        # ════════════════════════════════════════════════════════════════════
        '3L_A': _make_model([128, 64, 32]),   # 784 → 128 → 64 → 32 → 5
        '3L_B': _make_model([256, 64, 16]),   # 784 → 256 → 64 → 16 → 5

        # ════════════════════════════════════════════════════════════════════
        # 4 Hidden Layers  (2 architectures)
        # ════════════════════════════════════════════════════════════════════
        '4L_A': _make_model([256, 128, 32, 16]),  # 784 → 256 → 128 → 32 → 16 → 5
        '4L_B': _make_model([128,  64, 32, 16]),  # 784 → 128 →  64 → 32 → 16 → 5

        # ════════════════════════════════════════════════════════════════════
        # 5 Hidden Layers  (1 architecture)
        # ════════════════════════════════════════════════════════════════════
        '5L_A': _make_model([256, 128, 64, 32, 16]),  # 784 → 256 → 128 → 64 → 32 → 16 → 5
        '5L_B': _make_model([512,256, 128, 64, 32])
    }
