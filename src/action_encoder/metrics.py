"""Metric functions for ActionTextEncoder"""

import torch


def calculate_per_dimension_mae(encoder, decoder, dataloader):
    """
    Calculate Mean Absolute Error per dimension.

    Args:
        encoder: TextEncoder model
        decoder: TextDecoder model
        dataloader: DataLoader for evaluation

    Returns:
        Tensor of MAE per dimension
    """
    encoder.eval()
    decoder.eval()

    sample_data, _ = next(iter(dataloader))
    per_dim_mae = torch.zeros(sample_data.shape[1])
    n = 0

    with torch.no_grad():
        for data, _ in dataloader:
            rec = decoder(encoder(data))
            per_dim_mae += (rec - data).abs().mean(dim=0)
            n += 1

    return per_dim_mae / n
