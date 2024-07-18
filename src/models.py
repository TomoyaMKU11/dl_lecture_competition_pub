import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,
            seq_len: int,
            in_channels: int,
            hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_  
        Args:            X ( b, c, t ): _description_        Returns:            X ( b, num_classes ): _description_        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel_size: int = 3,
            p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)
        return self.dropout(X)


class BasicBiLSTMClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,
            seq_len: int,
            in_channels: int,
            hid_dim: int = 128,
            num_layers: int = 2,
            p_drop: float = 0.1
    ) -> None:
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=p_drop
        )

        self.head = nn.Sequential(
            nn.Linear(hid_dim * 2, num_classes)  # hid_dim * 2 because of bidirectional
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_  
        Args:            X ( b, c, t ): _description_        Returns:            X ( b, num_classes ): _description_        """  # X: (b, c, t) -> (b, t, c) for LSTM        X = X.permute(0, 2, 1)

        # LSTM forward pass
        X, _ = self.bilstm(X)

        # Take the output of the last time step
        X = X[:, -1, :]  # (b, hid_dim * 2)

        return self.head(X)


class BasicConvAttClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int,
            seq_len: int,
            in_channels: int,
            hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim)
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=4)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)

        # Ensure input shape is compatible with MultiheadAttention
        X = X.permute(2, 0, 1)  # Change shape to (seq_len, batch_size, hid_dim)

        # Apply attention mechanism        X, _ = self.attention(X, X, X)

        # Change shape back to (batch_size, seq_len, hid_dim)
        X = X.permute(1, 2, 0)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            kernel_size: int = 3,
            p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))
        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)

    class BasicConvTraClassifier(nn.Module):
        def __init__(
                self,
                num_classes: int,
                seq_len: int,
                in_channels: int,
                hid_dim: int = 128,
                nhead: int = 8,
                num_transformer_layers: int = 1
        ) -> None:
            super().__init__()

            self.blocks = nn.Sequential(
                ConvBlock(in_channels, hid_dim),
                ConvBlock(hid_dim, hid_dim),
            )

            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hid_dim,
                    nhead=nhead,
                    dim_feedforward=hid_dim * 2,
                    dropout=0.1,
                    activation='gelu'
                ),
                num_layers=num_transformer_layers
            )

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                Rearrange("b d 1 -> b d"),
                nn.Linear(hid_dim, num_classes),
            )

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            """_summary_  
            Args:                X ( b, c, t ): _description_            Returns:                X ( b, num_classes ): _description_            """
            X = self.blocks(X)

            X = X.permute(2, 0, 1)  # Transformerは入力を(seq_len, batch_size, hid_dim)の形で期待する
            X = self.transformer(X)
            X = X.permute(1, 2, 0)  # 元の形(batch_size, hid_dim, seq_len)に戻す

            return self.head(X)

    class ConvBlock(nn.Module):
        def __init__(
                self,
                in_dim,
                out_dim,
                kernel_size: int = 3,
                p_drop: float = 0.1,
        ) -> None:
            super().__init__()

            self.in_dim = in_dim
            self.out_dim = out_dim

            self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
            self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

            self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
            self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

            self.dropout = nn.Dropout(p_drop)

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            if self.in_dim == self.out_dim:
                X = self.conv0(X) + X  # スキップ接続
            else:
                X = self.conv0(X)

            X = F.gelu(self.batchnorm0(X))

            X = self.conv1(X) + X  # スキップ接続
            X = F.gelu(self.batchnorm1(X))

            return self.dropout(X)