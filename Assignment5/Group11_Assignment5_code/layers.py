"""
layers.py - the A2 ConvReLUPool class and every torch model built from it.

Part A's CNN (A3) and the three Caltech architectures (B1) are all stacks of
ConvReLUPool blocks followed by Flatten -> Linear+ReLU -> Linear (softmax is
applied by the loss / at prediction time).

Initialisation: conv kernels Kaiming normal (fan_in, ReLU gain), FC layers
Xavier normal, all biases zero.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_manual import out_size


class ConvReLUPool(nn.Module):
    """Convolution -> ReLU -> (optional) max pooling, with its own weights.

    `probe = True` swaps ReLU for identity and max pooling for average pooling
    of the same size and stride. The network is then linear with every window
    position connected, so the non-zero support of a gradient equals the
    receptive field exactly - used only by the smoke test.
    """

    def __init__(self, in_ch, out_ch, k, stride=1, pad=0,
                 pool_k=2, pool_s=2, pool=True):
        super().__init__()
        self.in_ch, self.out_ch, self.k = in_ch, out_ch, k
        self.stride, self.pad = stride, pad
        self.pool, self.pool_k, self.pool_s = pool, pool_k, pool_s
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        self.probe = False

    def conv_out(self, x):
        """Post-ReLU, pre-pool feature maps ("last conv layer" output)."""
        z = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.pad)
        return z if self.probe else F.relu(z)

    def pool_out(self, a):
        if not self.pool:
            return a
        if self.probe:
            return F.avg_pool2d(a, self.pool_k, self.pool_s)
        return F.max_pool2d(a, self.pool_k, self.pool_s)

    def forward(self, x):
        return self.pool_out(self.conv_out(x))

    def specs(self):
        """(k, s, p) of the conv, and of the pool if present."""
        conv = [(self.k, self.stride, self.pad)]
        return conv, ([(self.pool_k, self.pool_s, 0)] if self.pool else [])

    def out_hw(self, h):
        """(conv size, block output size) by the formula."""
        c = out_size(h, self.k, self.pad, self.stride)
        return c, (out_size(c, self.pool_k, 0, self.pool_s) if self.pool else c)

    def extra_repr(self):
        s = "%d->%d, %dx%d s%d p%d" % (self.in_ch, self.out_ch, self.k, self.k,
                                       self.stride, self.pad)
        if self.pool:
            s += ", maxpool %dx%d s%d" % (self.pool_k, self.pool_k, self.pool_s)
        return s


class CNN(nn.Module):
    """ConvReLUPool blocks -> flatten -> FC(hidden)+ReLU -> FC(n_classes)."""

    def __init__(self, blocks, in_ch, in_hw, hidden, n_classes):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.in_ch, self.in_hw = in_ch, in_hw
        h = in_hw
        for b in self.blocks:
            _, h = b.out_hw(h)
        self.flat_dim = self.blocks[-1].out_ch * h * h
        self.fc1 = nn.Linear(self.flat_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        for fc in (self.fc1, self.fc2):
            nn.init.xavier_normal_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))

    def features(self, x):
        """Per block: {"conv": post-ReLU pre-pool maps, "out": block output}."""
        feats = []
        for b in self.blocks:
            a = b.conv_out(x)
            x = b.pool_out(a)
            feats.append({"conv": a, "out": x})
        return feats

    def conv_specs(self, block_idx):
        """Layer specs from the input up to the conv output of block `block_idx`."""
        specs = []
        for i, b in enumerate(self.blocks[:block_idx + 1]):
            conv, pool = b.specs()
            specs += conv
            if i < block_idx:
                specs += pool
        return specs

    def size_table(self):
        """Formula sizes per stage: [(name, channels, h, w)]."""
        rows = [("input", self.in_ch, self.in_hw, self.in_hw)]
        h = self.in_hw
        for i, b in enumerate(self.blocks, 1):
            c, o = b.out_hw(h)
            rows.append(("conv%d" % i, b.out_ch, c, c))
            if b.pool:
                rows.append(("pool%d" % i, b.out_ch, o, o))
            h = o
        rows += [("flatten", self.flat_dim, 1, 1),
                 ("fc1", self.fc1.out_features, 1, 1),
                 ("fc2", self.fc2.out_features, 1, 1)]
        return rows

    def set_probe(self, on):
        for b in self.blocks:
            b.probe = on


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- Part A

def part_a_blocks(seed):
    """A2's two stacked layers: 6@5x5 + pool 2/2, 16@5x5 + pool 2/2."""
    torch.manual_seed(seed)
    return [ConvReLUPool(1, 6, 5, pool_k=2, pool_s=2),
            ConvReLUPool(6, 16, 5, pool_k=2, pool_s=2)]


def part_a_cnn(blocks, seed):
    """A3's CNN, starting from (a copy of) the A2 blocks so the trained
    filters can be compared with the untrained ones at the same index."""
    blocks = copy.deepcopy(blocks)
    torch.manual_seed(seed + 1)      # FC init; conv weights come from A2
    return CNN(blocks, in_ch=1, in_hw=28, hidden=64, n_classes=5)


# ---------------------------------------------------------------- Part B

ARCH_NAMES = {1: "Arch-1 (4-layer)", 2: "Arch-2 (5-layer)", 3: "Arch-3 (6-layer)"}


def caltech_arch(n, seed):
    """Architectures 1-3. conv1/conv2 are created first from the same seed, so
    all three start with identical weights in the layers they share."""
    torch.manual_seed(seed)
    blocks = [ConvReLUPool(3, 8, 11, stride=4, pool_k=3, pool_s=2),
              ConvReLUPool(8, 16, 5, pool_k=3, pool_s=2)]
    if n == 2:
        blocks.append(ConvReLUPool(16, 32, 3, pool_k=3, pool_s=2))
    elif n == 3:
        blocks.append(ConvReLUPool(16, 32, 3, pool=False))
        blocks.append(ConvReLUPool(32, 64, 3, pool_k=3, pool_s=2))
    elif n != 1:
        raise ValueError("architecture %r" % n)
    return CNN(blocks, in_ch=3, in_hw=224, hidden=128, n_classes=5)
