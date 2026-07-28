from torch.utils.data import Dataset


class NextFrameDataset(Dataset):
    def __init__(self, data_files):
        self.X = []
        self.y = []

        for file_data in data_files:
            x = file_data["x"]
            y = file_data["y"]

            for t in range(x.shape[0]):
                self.X.append(x[t].float())
                self.y.append(y[t].float())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
