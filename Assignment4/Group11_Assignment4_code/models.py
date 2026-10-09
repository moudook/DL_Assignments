"""
models.py — FCNN classifiers and autoencoders for Assignment-4.

Activation policy (locked)
--------------------------
Logistic sigmoid everywhere. A4's statement names sigmoid once (autoencoder hidden
layers) and permits logistic or tanh provided one is used consistently; the
instruction not to use tanh unless explicitly mentioned resolves that to logistic.
Tanh would additionally force targets into [-1,1], which puts reconstruction error
out of pixel units and complicates every number reported in Task-2c and Task-5b.

Autoencoder output is sigmoid so reconstructions live in [0,1], matching the
normalised targets. The bottleneck (middle) layer is always LINEAR, no activation,
as mandated by A4.

Classifier activation is sigmoid for the same consistency reason. Note this
differs from Assignment-3, whose classifiers used tanh; A4 does not require
classifier activation parity, and the report will state the choice.

Weight init
-----------
Xavier Glorot uniform, zero bias, matching Assignment-3's models.py. torch.manual_seed
is set once per model build so every architecture draws a reproducible slice of the
stream. The 4-hidden and 5-hidden shapes suit Xavier (tanh-like symmetric regime);
sigmoid is also fine with Xavier but slightly conservative.
"""

import torch
import torch.nn as nn

INIT_SEED = 42

CLASSIFIER_ARCHS = {
    "3L_A": [128, 64, 32],
    "3L_B": [64, 32, 16],
    "4L_A": [128, 64, 32, 16],
    "5L_A": [64, 32, 16, 8, 4],
}

BOTTLENECKS = [32, 64, 128, 256]

AE_OUTER = 400


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class FCNN(nn.Module):
    """
    Configurable fully connected classifier.

    input_dim varies by task (784 for raw, 32/64/128/256 for reduced
    representations). Hidden layers use sigmoid; the output layer emits raw
    logits and CrossEntropyLoss applies log-softmax internally.
    """

    def __init__(self, input_dim=784, hidden_sizes=(128, 64, 32), num_classes=5):
        super().__init__()
        layers, cur = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(cur, h), nn.Sigmoid()]
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x)


def build_classifier(arch_name, input_dim=784, num_classes=5):
    """Build one named classifier with a fresh seeded init."""
    torch.manual_seed(INIT_SEED)
    return FCNN(input_dim=input_dim, hidden_sizes=CLASSIFIER_ARCHS[arch_name],
                num_classes=num_classes)


def build_all_classifiers(input_dim, num_classes=5, archs=None):
    """
    Build every classifier for a given input dimension.

    Seeded once for the whole dict so the set is reproducible as a unit, matching
    how Assignment-3's get_models() behaved.
    """
    names = list(archs or CLASSIFIER_ARCHS)
    torch.manual_seed(INIT_SEED)
    return {n: FCNN(input_dim=input_dim,
                     hidden_sizes=CLASSIFIER_ARCHS[n],
                     num_classes=num_classes) for n in names}


class Autoencoder(nn.Module):
    """
    Autoencoder over 784-d flattened images.

    kind:
        "1hidden" -> 784 -> k          (bottleneck is the only hidden layer)
        "3hidden" -> 784 -> 400 -> k -> 400 -> 784

    k is the linear bottleneck (no activation), per A4. All other hidden layers
    use sigmoid; the output layer is sigmoid so reconstructions are in [0,1].

    For "1hidden", the single hidden layer IS the bottleneck, so it is linear and
    there are no sigmoid hidden layers. That is the literal reading of A4: the
    middle layer is always linear, and sigmoid applies to remaining hidden layers,
    of which there are none.
    """

    def __init__(self, bottleneck=32, kind="1hidden", outer=AE_OUTER,
                 input_dim=784):
        super().__init__()
        self.kind = kind
        self.bottleneck = bottleneck

        if kind == "1hidden":
            self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck))
            self.decoder = nn.Sequential(nn.Linear(bottleneck, input_dim),
                                         nn.Sigmoid())
        elif kind == "3hidden":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, outer), nn.Sigmoid(),
                nn.Linear(outer, bottleneck),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, outer), nn.Sigmoid(),
                nn.Linear(outer, input_dim), nn.Sigmoid(),
            )
        else:
            raise ValueError(f"unknown kind {kind!r}; use '1hidden' or '3hidden'")

        self.apply(_init_weights)

    def encode(self, x):
        """Compressed representation — what Tasks 3, 4, 5 and 6 consume."""
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encode(x))


def build_autoencoder(kind="1hidden", bottleneck=32, input_dim=784, seed=INIT_SEED):
    """Build an autoencoder with a fresh seeded init."""
    torch.manual_seed(seed)
    return Autoencoder(bottleneck=bottleneck, kind=kind, input_dim=input_dim)


def build_all_autoencoders(bottlenecks=BOTTLENECKS, input_dim=784):
    """Every required (kind, bottleneck) autoencoder, seeded once as a set."""
    out = {}
    for kind in ("1hidden", "3hidden"):
        for b in bottlenecks:
            out[f"{kind}_{b}"] = Autoencoder(bottleneck=b, kind=kind,
                                             input_dim=input_dim)
    return out


def build_denoising_autoencoder(bottleneck, noise=0.2, input_dim=784):
    """
    1-hidden denoising autoencoder for A4 Task-5.

    Structurally identical to the 1-hidden autoencoder. The difference is in
    training, not architecture: inputs are corrupted before the forward pass and
    the RECONSTRUCTION TARGET is the clean image. That is what forces the
    bottleneck to discard noise rather than memorize it.

    `noise` is the corruption fraction (0.2 or 0.4 per A4). It is stored on the
    module only so the checkpoint and the report both record which noise level a
    given model used; corruption itself is applied in training.
    """
    torch.manual_seed(INIT_SEED)
    model = Autoencoder(bottleneck=bottleneck, kind="1hidden",
                        input_dim=input_dim)
    model.noise_level = noise
    return model


def make_noise(x, level, generator=None):
    """
    Masking corruption: with probability `level`, zero out an input pixel.

    Masking (rather than additive Gaussian) is the standard denoising-AE choice
    for images: it models salt-like pixel corruption and keeps values in [0,1],
    so the target distribution is unchanged and the sigmoid output layer stays
    well-matched. Applied per-sample independently, so every image gets its own
    random mask pattern rather than one shared pattern per batch.

    `generator` makes the noise reproducible; pass a seeded generator for
    run-to-run reproducibility.
    """
    if level <= 0:
        return x
    keep = torch.rand(x.shape, device=x.device, generator=generator) >= level
    return x * keep.float()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def describe(model):
    """Shape string, for the report's architecture tables."""
    return " -> ".join(str(m.out_features) if isinstance(m, nn.Linear)
                       else "sigmoid"
                       for m in model.net) if isinstance(getattr(model, "net", None),
                                                        nn.Sequential) else "custom"


def clone_state(model):
    """Detached state_dict copy, cheaper than copy.deepcopy for reset purposes."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
