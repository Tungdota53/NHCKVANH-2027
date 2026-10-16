"""Adaptive average cells without the nondeterministic CUDA pooling backward."""


def deterministic_adaptive_pool2d(inputs, output_size):
    import torch
    if inputs.ndim != 4:
        raise ValueError("Research pooling expects NCHW")
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    height, width = inputs.shape[-2:]
    oh, ow = height if oh is None else oh, width if ow is None else ow
    if min(oh, ow) < 1:
        raise ValueError("Invalid adaptive pooling output size")
    rows = []
    for i in range(oh):
        top, bottom = i * height // oh, ((i + 1) * height + oh - 1) // oh
        cells = []
        for j in range(ow):
            left, right = j * width // ow, ((j + 1) * width + ow - 1) // ow
            cells.append(inputs[..., top:bottom, left:right].mean(dim=(-2, -1)))
        rows.append(torch.stack(cells, dim=-1))
    return torch.stack(rows, dim=-2)
