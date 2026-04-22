"""Training utility helpers."""


def count_trainable_parameters(model) -> int:
    """Return the number of trainable parameters for a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
