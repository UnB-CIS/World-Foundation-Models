"""Encoding function for Action Encoder."""

import torch


def encoding_function(
    type_encoding: dict,
    object_encoding: dict,
    action_data: dict | None = None,
    screen_width: int = 800,
    screen_height: int = 600,
    input_vector_dim: int = 4,
) -> torch.Tensor:
    """Convert an action dictionary or 'no action' signal to a tensor.

    Args:
        type_encoding (dict): One-hot encoding for action type.
        object_encoding (dict): One-hot encoding for object type.
        action_data (dict | None, optional): Action dictionary (JSON) or None for no action. Defaults to None.
        screen_width (int, optional): Screen width for scenario. Defaults to 800 (scenario 1).
        screen_height (int, optional): Screen height for scenario. Defaults to 600 (scenario 2).
        input_vector_dim (int, optional): Dimension of vector. Defaults to 4.

    Raises:
        ValueError: Raises error if list length is different from input_vector_dim.

    Returns:
        torch.Tensor: PyTorch tensor of dimension [INPUT_VECTOR_DIM]
    """
    type_vector = object_vector = []
    x_norm = y_norm = 0.0

    # No action case
    if action_data is None:
        type_vector = type_encoding['none']
        object_vector = object_encoding['none']
        x_norm = 0.0
        y_norm = 0.0

    # Positive action (mouse_down)
    elif action_data:
        type_vector = type_encoding['mouse_down']
        object_vector = object_encoding.get(action_data['object'], [0.0])
        pos_x = action_data['pos'][0]
        pos_y = action_data['pos'][1]

        # Normalize screen coordinates to [0, 1]
        x_norm = pos_x / screen_width
        y_norm = pos_y / screen_height

    input_list = type_vector + object_vector + [x_norm, y_norm]

    if len(input_list) != input_vector_dim:
        raise ValueError(f"Invalid vector dimension: {len(input_list)}")

    return torch.tensor(input_list, dtype=torch.float32)
