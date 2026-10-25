"""
conv_manual.py - A1: convolution by explicit traversal, and the output-size
formula used to check every layer in the assignment.

The traversal is cross-correlation (no kernel flip): s[i, j] =
sum_a sum_b I[i*S + a, j*S + b] * K[a, b], the notes' image formula and what
PyTorch/Keras compute. For a random filter the flip changes nothing anyway.
"""

import numpy as np


def out_size(n, f, p=0, s=1):
    """floor((n - f + 2p) / s) + 1.

    The notes print the formula without the floor; frameworks floor, and the
    Caltech layers need it ((224 - 11) / 4 = 53.25).
    """
    return (n - f + 2 * p) // s + 1


def kaiming_filter(shape, fan_in, rng):
    """N(0, 2 / fan_in): Kaiming normal, fan_in mode, ReLU gain."""
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


def conv2d_manual(image, kernel, stride=1, pad=0):
    """Slide `kernel` (F x F) over every valid position of `image` (H x W)."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if pad:
        image = np.pad(image, pad)
    H, W = image.shape
    F = kernel.shape[0]
    Ho = (H - F) // stride + 1
    Wo = (W - F) // stride + 1
    out = np.zeros((Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            r, c = i * stride, j * stride
            acc = 0.0
            for a in range(F):
                for b in range(F):
                    acc += image[r + a, c + b] * kernel[a, b]
            out[i, j] = acc
    return out
