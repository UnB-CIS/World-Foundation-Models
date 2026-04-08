"""Training Dataset for LatentDecoder."""

import torch
from torch.utils.data.dataset import Dataset


class LatentDataset(Dataset):
    def __init__(self, data_files: list[dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self.X = []
        self.y = []

        for file_data in data_files:
            x = file_data["x"]
            y = file_data["y"]

            for t in range(x.shape[0]):
                self.X.append(x[t])
                self.y.append(y[t])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]
