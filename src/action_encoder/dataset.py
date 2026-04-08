"""Dataset generation for Action Encoder."""

import torch
import random

from .encoding import encoding_function


def generate_synthetic_dataset(
    num_samples: int,
    type_encoding: dict,
    object_encoding: dict,
    action_data: dict | None = None,
    screen_width: int = 800,
    screen_height: int = 600,
    input_vector_dim: int = 4,
) -> torch.Tensor:
    """Generate synthetic dataset with 50% action and 50% no-action samples.

    Args:
        num_samples: Total number of samples
        type_encoding (dict): One-hot encoding for action type.
        object_encoding (dict): One-hot encoding for object type.
        action_data (dict | None, optional): Action dictionary (JSON) or None for no action. Defaults to None.
        screen_width (int, optional): Screen width for scenario. Defaults to 800 (scenario 1).
        screen_height (int, optional): Screen height for scenario. Defaults to 600 (scenario 2).
        input_vector_dim (int, optional): Dimension of vector. Defaults to 4.

    Returns:
        Tensor of encoded actions
    """
    synthetic_inputs = []

    # Generate action samples (50%)
    for _ in range(num_samples // 2):
        pos_x = random.randint(50, 750)
        pos_y = random.randint(50, 500)

        action_data = {"type": "mouse_down", "object": "ball", "pos": [pos_x, pos_y]}

        synthetic_inputs.append(
            encoding_function(
                type_encoding=type_encoding,
                object_encoding=object_encoding,
                action_data=action_data,
                screen_width=screen_width,
                screen_height=screen_height,
                input_vector_dim=input_vector_dim,
            )
        )

    # Generate no-action samples (50%)
    for _ in range(num_samples // 2):
        synthetic_inputs.append(
            encoding_function(
                type_encoding=type_encoding,
                object_encoding=object_encoding,
                screen_width=screen_width,
                screen_height=screen_height,
                input_vector_dim=input_vector_dim,
            )
        )

    return torch.stack(synthetic_inputs)
