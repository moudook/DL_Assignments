"""
receptive.py - trace a neuron back to the input patch that can affect it.

This is the notes' procedure (slides 111-115): from the neuron, step back one
layer at a time, widening the window by that layer's kernel and stride, until
the input. Whole pooling windows are kept, not only the argmax path, as in the
slides.

A layer is (k, s, p): kernel size, stride, padding - the same for conv and
pool. Boxes are inclusive (r0, r1, c0, c1) in input pixel coordinates.
"""


def trace_box(specs, i, j):
    """Unclipped input box of output position (i, j) after `specs`."""
    r0 = r1 = i
    c0 = c1 = j
    for k, s, p in reversed(specs):
        r0, r1 = r0 * s - p, r1 * s - p + k - 1
        c0, c1 = c0 * s - p, c1 * s - p + k - 1
    return r0, r1, c0, c1


def clip_box(box, h, w):
    r0, r1, c0, c1 = box
    return max(r0, 0), min(r1, h - 1), max(c0, 0), min(c1, w - 1)


def rf_size_and_jump(specs):
    """Receptive-field side length and input-pixel step between neighbours."""
    rf, jump = 1, 1
    for k, s, _ in specs:
        rf += (k - 1) * jump
        jump *= s
    return rf, jump


def neuron_box(specs, i, j, h, w):
    """Clipped box plus whether clipping happened (padding / RF > image)."""
    raw = trace_box(specs, i, j)
    box = clip_box(raw, h, w)
    return box, box != raw
