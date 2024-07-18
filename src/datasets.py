import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        # データの読み込み
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))  # .to('cuda')
        # ベースライン処理
        baseline_mean = self.X.mean(dim=2, keepdim=True)
        self.X -= baseline_mean
        # ロバストスケーラー
        median = self.X.median(dim=2, keepdim=True)[0]
        q1 = self.X.quantile(0.25, dim=2, keepdim=True)
        q3 = self.X.quantile(0.75, dim=2, keepdim=True)
        iqr = q3 - q1

        # ロバストスケーリングを適用
        self.X = (self.X - median) / iqr

        # [-20, 20]の範囲外の値をクリップ
        self.X = torch.clamp(self.X, -20, 20)
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))  # .to('cuda')

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))  # .to('cuda')
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]