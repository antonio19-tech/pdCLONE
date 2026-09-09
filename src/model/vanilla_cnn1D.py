import torch
from torch import nn


class VanillaCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embedding_dim: int = 64,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2")

        self.features = nn.Sequential(
            # FORMULA = L(out) = ⌊(L(in) + 2P − K) / S⌋ + 1
            # dove :
            # L(in) -> lunghezza della sequenza in input;
            # P -> padding, ovvero la sequenza viene allungata sia a destra
            # che sinistra con degli 0, in modo da permettere al filtro di scorrere;
            # K -> kernel_size, stabilisce il numero di elementi utilizzati
            # dal filtro;
            # S -> stride, indica di quanto si sposta il filtro ogni volta.

            # Il filtro in una cnn serve per riconoscere pattern nella sequenza,
            # è un vettore di pesi: w = [w1, w2, ... , wk].
            # Questo vettore di pesi (filtro convoluzionale) viene condiviso tra tutte le sequenze
            # in input e aggiornato durante il training tramite backpropagation.
            # In questo modo la rete apprende pattern generali che possono essere
            # riutilizzati su tutte le sequenze o segmenti temporali.

            # La formula è: y[i] somm(da k=0 a k-1) x[i+k]w[k];
            # dove:
            # x -> sequenza in entrata;
            # w -> filtro/kernel
            # K = dimensione del filtro
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(8, 32),  # Normalizza le feature
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),  # Riduce la dimensione temporale

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

        # Preserve a small temporal structure instead of collapsing everything to 1
        self.global_pool = nn.AdaptiveAvgPool1d(4) # Riduce sempre la sequenza a lunghezza 4.

        # latent projection
        self.embedding = nn.Sequential(
            nn.Linear(128 * 4, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        # final classifier
        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None,
                return_embeddings: bool = False,
                ) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        returns: (batch, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, seq_len, features), got {tuple(x.shape)}")

        x = x.transpose(1, 2)  # (batch, features, seq_len)

        if mask is not None:

            mask = mask.float()

            mask = mask[:, None, :]

            mask = nn.functional.interpolate(
                mask,
                size=x.shape[-1],
                mode="nearest",
            )

            x = x * mask

        x = self.features(x)
        x = self.global_pool(x)  # (batch, 128, 4)
        x = torch.flatten(x, start_dim=1)  # (batch, 128 * 4)
        embeddings = self.embedding(x)

        if return_embeddings:
            return embeddings
        logit = self.classifier(embeddings)

        return logit
