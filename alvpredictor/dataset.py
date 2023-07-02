import pickle

import numpy as np
from torch.utils.data import Dataset


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


class Dataset(Dataset):
    def __init__(
        self, mode, preprocess_config,
    ):
        assert mode in ["train", "val"]
        self.data_dir = preprocess_config["path"]["data_dir"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]

        num_class = preprocess_config["preprocessing"]["num_class"]
        with open(f'{self.data_dir}/text_{num_class}.bin', 'rb') as p:
            self.texts = pickle.load(p)
        with open(f'{self.data_dir}/alv_{num_class}.bin', 'rb') as p:
            self.alvs = pickle.load(p)
        if mode == "train":
            self.texts = self.texts[self.val_size:]
            self.alvs = self.alvs[self.val_size:]
        if mode == "val":
            self.texts = self.texts[:self.val_size]
            self.alvs = self.alvs[:self.val_size]

        self.text_lens = np.array([alv.shape[0] for alv in self.alvs])
        self.texts = pad_1D(self.texts)
        self.alvs = pad_1D(self.alvs)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sample = {
            "text": self.texts[idx],
            "alv": self.alvs[idx],
            "text_lens": self.text_lens[idx]
        }

        return sample


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("alvpredictor/config/JSUT/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("alvpredictor/config/JSUT/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train", preprocess_config
    )
    val_dataset = Dataset(
        "val", preprocess_config
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
    print(train_dataset[0]["text"].shape)
    print(train_dataset[0]["alv"].shape)
    print(train_dataset[0]["text"])
    print(train_dataset[0]["alv"])
