"""
Text Encoder for Action Encoding

This module implements a deterministic encoder and decoder for actions in a simulation.
Each action contains:
- time: timestamp in seconds
- type: action type (mouse_down)
- object: affected object (ball)
- pos: [x, y] coordinates of the action

Note: The 'time' field is not encoded - temporal alignment is handled separately.
"""

import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# =============================================================================
# Configuration Constants
# =============================================================================

# Screen dimensions for scenario 1
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# One-hot encoding for action type
TYPE_ENCODING = {
    'mouse_down': [1.0],  # Action present
    'none': [0.0]         # No action
}

# One-hot encoding for object type
OBJECT_ENCODING = {
    'ball': [1.0],        # Scenario 1 only has balls
    'none': [0.0]         # No action - zero vector
}

# Vector dimensions
INPUT_VECTOR_DIM = 4      # [type, object, x_norm, y_norm]
LATENT_ACTION_DIM = 16    # Za: Latent action space dimension

# Dataset configuration
NUM_SAMPLES = 2000
BATCH_SIZE = 64
VAL_RATIO = 0.2           # 20% for validation
NUM_EPOCHS = 20

# Set random seed for reproducibility
torch.manual_seed(42)


# =============================================================================
# Encoding Functions
# =============================================================================

def encoding_function(action_data: dict = None) -> torch.Tensor:
    """
    Convert an action dictionary or 'no action' signal to a tensor.
    
    Args:
        action_data: Action dictionary (JSON) or None for no action
        
    Returns:
        PyTorch tensor of dimension [INPUT_VECTOR_DIM]
    """
    # No action case
    if action_data is None:
        type_vector = TYPE_ENCODING['none']
        object_vector = OBJECT_ENCODING['none']
        x_norm = 0.0
        y_norm = 0.0

    # Positive action (mouse_down)
    elif action_data:
        type_vector = TYPE_ENCODING['mouse_down']
        object_vector = OBJECT_ENCODING.get(action_data['object'], [0.0])
        pos_x = action_data['pos'][0]
        pos_y = action_data['pos'][1]

        # Normalize screen coordinates to [0, 1]
        x_norm = pos_x / SCREEN_WIDTH
        y_norm = pos_y / SCREEN_HEIGHT

    input_list = type_vector + object_vector + [x_norm, y_norm]

    if len(input_list) != INPUT_VECTOR_DIM:
        raise ValueError(f"Invalid vector dimension: {len(input_list)}")

    return torch.tensor(input_list, dtype=torch.float32)


# =============================================================================
# Model Architectures
# =============================================================================

class TextEncoder(nn.Module):
    """
    Deterministic encoder that maps input vector to latent space.
    """
    
    def __init__(self, input_dim=INPUT_VECTOR_DIM, latent_dim=LATENT_ACTION_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class TextDecoder(nn.Module):
    """
    Deterministic decoder that reconstructs action vector from latent space.
    """
    
    def __init__(self, latent_dim=LATENT_ACTION_DIM, output_dim=INPUT_VECTOR_DIM):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, z):
        x = self.net(z)
        return torch.sigmoid(x)


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_synthetic_dataset(num_samples=NUM_SAMPLES):
    """
    Generate synthetic dataset with 50% action and 50% no-action samples.
    
    Args:
        num_samples: Total number of samples
        
    Returns:
        Tensor of encoded actions
    """
    synthetic_inputs = []

    # Generate action samples (50%)
    for _ in range(num_samples // 2):
        pos_x = random.randint(50, 750)
        pos_y = random.randint(50, 500)

        action_data = {
            "type": "mouse_down",
            "object": "ball",
            "pos": [pos_x, pos_y]
        }

        synthetic_inputs.append(encoding_function(action_data))

    # Generate no-action samples (50%)
    for _ in range(num_samples // 2):
        synthetic_inputs.append(encoding_function(None))

    return torch.stack(synthetic_inputs)


# =============================================================================
# Evaluation Utilities
# =============================================================================

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


def test_single_actions(encoder, decoder):
    """
    Test reconstruction of individual actions.
    
    Args:
        encoder: TextEncoder model
        decoder: TextDecoder model
    """
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Test 1 - No Action
        v_none_original = encoding_function(None)
        z_none = encoder(v_none_original.unsqueeze(0))
        reconstruction_none = decoder(z_none).squeeze(0)

        print("Test 1: No Action Reconstruction")
        print(f"Original:     {v_none_original.tolist()}")
        print(f"Reconstructed: {[round(x, 4) for x in reconstruction_none.tolist()]}")
        print()

        # Test 2 - Mouse Down Action
        pos_x, pos_y = 640, 120
        test_action = {
            "type": "mouse_down",
            "object": "ball",
            "pos": [pos_x, pos_y]
        }

        v_down_original = encoding_function(test_action)
        z_down = encoder(v_down_original.unsqueeze(0))
        reconstruction_down = decoder(z_down).squeeze(0)

        x_expected = pos_x / SCREEN_WIDTH
        y_expected = pos_y / SCREEN_HEIGHT

        print("Test 2: Mouse Down Action Reconstruction")
        print(f"Original:     {v_down_original.tolist()}")
        print(f"Reconstructed: {[round(x, 4) for x in reconstruction_down.tolist()]}")
        print()

        print("Position components (normalized):")
        print(f"Original:     x={x_expected:.4f}, y={y_expected:.4f}")
        print(f"Reconstructed: x={reconstruction_down[2]:.4f}, y={reconstruction_down[3]:.4f}")

        max_error = torch.abs(reconstruction_down - v_down_original).max().item()
        print(f"\nMax absolute error: {max_error:.6f}")


# =============================================================================
# Training Function
# =============================================================================

def train_models(encoder, decoder, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """
    Train encoder and decoder models.
    
    Args:
        encoder: TextEncoder model
        decoder: TextDecoder model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        num_epochs: Number of training epochs
        
    Returns:
        train_losses, val_losses lists
    """
    # Combine parameters from both models
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    loss_function = nn.MSELoss()

    train_losses = []
    val_losses = []

    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training
        decoder.train()
        encoder.train()
        total_loss = 0

        for data, _ in train_loader:
            optimizer.zero_grad()

            # Forward pass
            z_a = encoder(data)
            reconstruction = decoder(z_a)

            # Calculate loss
            loss = loss_function(reconstruction, data)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss_total = 0

        with torch.no_grad():
            for data, _ in val_loader:
                z_a = encoder(data)
                reconstruction = decoder(z_a)
                val_loss = loss_function(reconstruction, data)
                val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {avg_train_loss:.6f}")
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    return train_losses, val_losses


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_training_history(train_losses, val_losses):
    """
    Plot training and validation losses.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training", marker='o')
    plt.plot(val_losses, label="Validation", marker='s')
    plt.title("Reconstruction Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def plot_with_best_epoch(train_losses, val_losses):
    """
    Plot training and validation losses with best epoch marked.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    best_epoch = int(np.argmin(val_losses))
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training", marker='o')
    plt.plot(val_losses, label="Validation", marker='s')
    plt.axvline(best_epoch, linestyle='--', alpha=0.6, label=f"Best @ {best_epoch}")
    plt.title("Reconstruction Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Text Encoder Training")
    print("=" * 70)
    
    # Test encoding functions
    example_action = {
        "time": 9.40,
        "type": "mouse_down",
        "object": "ball",
        "pos": [400, 300]
    }
    
    print("\nEncoding examples:")
    action_vector = encoding_function(example_action)
    print(f"Action present: {action_vector.tolist()}")
    
    none_vector = encoding_function(None)
    print(f"No action: {none_vector.tolist()}")
    
    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    full_tensor = generate_synthetic_dataset(NUM_SAMPLES)
    dataset = TensorDataset(full_tensor, full_tensor)
    print(f"Synthetic dataset generated: {len(full_tensor)} samples")
    print(f"Batch shape: {next(iter(DataLoader(dataset, batch_size=BATCH_SIZE)))[0].shape}")
    
    # Split into training and validation
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Initialize models
    text_encoder = TextEncoder()
    text_decoder = TextDecoder()
    
    print(f"\nEncoder params: {sum(p.numel() for p in text_encoder.parameters())}")
    print(f"Decoder params: {sum(p.numel() for p in text_decoder.parameters())}")
    
    # Train models
    train_losses, val_losses = train_models(
        text_encoder, text_decoder, train_loader, val_loader, NUM_EPOCHS
    )
    
    # Save models
    torch.save(text_encoder.state_dict(), "text_encoder_weights.pth")
    print("\nText Encoder weights saved!")
    
    # Plot results
    print("\nPlotting training history...")
    plot_training_history(train_losses, val_losses)
    plot_with_best_epoch(train_losses, val_losses)
    
    # Calculate per-dimension MAE
    per_dim_mae = calculate_per_dimension_mae(text_encoder, text_decoder, val_loader)
    print(f"\nMAE per dimension: {per_dim_mae.detach().cpu().numpy()}")
    print("  [type, object, x, y]")
    
    # Test single actions
    print("\n" + "=" * 50)
    print("Single Action Tests")
    print("=" * 50)
    test_single_actions(text_encoder, text_decoder)
    
    print("\nTraining complete!")