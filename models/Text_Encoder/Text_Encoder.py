#!/usr/bin/env python
# coding: utf-8

"""
Text Encoder Module
-------------------
Este módulo fornece:

• encoding_function → converte ação JSON em tensor
• TextEncoder → encoder determinístico
• TextDecoder → decoder determinístico
• funções opcionais de treino

Seguro para importação sem executar treinamento automaticamente.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import random
import numpy as np

# =========================================================
# CONFIGURAÇÕES
# =========================================================

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

TYPE_ENCODING = {
    "mouse_down": [1.0],
    "none": [0.0]
}

OBJECT_ENCODING = {
    "ball": [1.0],
    "none": [0.0]
}

INPUT_VECTOR_DIM = 4
LATENT_ACTION_DIM = 16


# =========================================================
# ENCODING FUNCTION
# =========================================================

def encoding_function(action_data: dict = None) -> torch.Tensor:
    """
    Converte ação JSON em tensor

    Output shape: [4]
    [type, object, x_norm, y_norm]
    """

    if action_data is None:

        type_vector = TYPE_ENCODING["none"]
        object_vector = OBJECT_ENCODING["none"]
        x_norm = 0.0
        y_norm = 0.0

    else:

        type_vector = TYPE_ENCODING["mouse_down"]

        object_vector = OBJECT_ENCODING.get(
            action_data.get("object", "none"),
            OBJECT_ENCODING["none"]
        )

        pos_x = action_data["pos"][0]
        pos_y = action_data["pos"][1]

        x_norm = pos_x / SCREEN_WIDTH
        y_norm = pos_y / SCREEN_HEIGHT

    input_list = type_vector + object_vector + [x_norm, y_norm]

    if len(input_list) != INPUT_VECTOR_DIM:
        raise ValueError(f"Dimensão inválida: {len(input_list)}")

    return torch.tensor(input_list, dtype=torch.float32)


# =========================================================
# MODELOS
# =========================================================

class TextEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = INPUT_VECTOR_DIM,
        latent_dim: int = LATENT_ACTION_DIM
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

    def forward(self, x):

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        return self.net(x)


class TextDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int = LATENT_ACTION_DIM,
        output_dim: int = INPUT_VECTOR_DIM
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):

        if len(z.shape) == 1:
            z = z.unsqueeze(0)

        return self.net(z)


# =========================================================
# DATASET SINTÉTICO
# =========================================================

def generate_synthetic_dataset(num_samples=2000):

    synthetic_inputs = []

    half = num_samples // 2

    # ações positivas
    for _ in range(half):

        pos_x = random.randint(50, 750)
        pos_y = random.randint(50, 550)

        action = {
            "type": "mouse_down",
            "object": "ball",
            "pos": [pos_x, pos_y]
        }

        synthetic_inputs.append(encoding_function(action))

    # ações nulas
    for _ in range(half):

        synthetic_inputs.append(encoding_function(None))

    tensor = torch.stack(synthetic_inputs)

    dataset = TensorDataset(tensor, tensor)

    return dataset


# =========================================================
# TREINAMENTO
# =========================================================

def train_text_encoder(
    epochs=20,
    batch_size=64,
    lr=1e-3,
    save_path="text_encoder_weights.pth"
):

    dataset = generate_synthetic_dataset()

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    encoder = TextEncoder()
    decoder = TextDecoder()

    params = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = optim.Adam(params, lr=lr)

    loss_fn = nn.MSELoss()

    print("Training Text Encoder...")

    for epoch in range(epochs):

        encoder.train()
        decoder.train()

        train_loss = 0

        for x, _ in train_loader:

            optimizer.zero_grad()

            z = encoder(x)

            recon = decoder(z)

            loss = loss_fn(recon, x)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        encoder.eval()
        decoder.eval()

        val_loss = 0

        with torch.no_grad():

            for x, _ in val_loader:

                z = encoder(x)

                recon = decoder(z)

                loss = loss_fn(recon, x)

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    torch.save(encoder.state_dict(), save_path)

    print(f"\nPesos salvos em: {save_path}")

    return encoder


# =========================================================
# LOAD
# =========================================================

def load_text_encoder(weights_path, device="cpu"):

    model = TextEncoder()

    state = torch.load(
        weights_path,
        map_location=device
    )

    model.load_state_dict(state)

    model.eval()

    return model


