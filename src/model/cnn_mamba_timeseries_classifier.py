from __future__ import annotations

import torch
from torch import nn

from src.model.vanilla_cnn1D import VanillaCNN1D
from src.model.mamba_sequence_classifier import MambaSequenceClassifier


class CNNMambaTimeSeriesClassifier(nn.Module):
    """
    CNN + Mamba classifier for complete time series.

    Input:
        x: [B, T, F]

    Output:
        logits: [B, num_classes]

    The CNN extracts local temporal features and downsamples
    the original sequence. Mamba then models the resulting
    sequence of CNN features.

    Nota: di VanillaCNN1D viene riusato SOLO l'estrattore convoluzionale.
    Il pooling globale, la proiezione latente e il classifier non vengono
    usati, perche' collasserebbero la dimensione temporale che Mamba deve
    invece scandire. Per questo motivo non esiste un parametro di embedding
    dimension: l'input di Mamba e' fissato dal numero di canali in uscita
    dall'ultima convoluzione.
    """

    CNN_OUTPUT_CHANNELS = 128

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        mamba_d_model: int = 128,
        mamba_layers: int = 2,
        mamba_state: int = 16,
        mamba_expand: int = 2,
        mamba_conv: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2")

        # We reuse the CNN architecture already present in the project,
        # but only its convolutional feature extractor.
        cnn = VanillaCNN1D(
            in_channels=input_dim,
            num_classes=num_classes,
        )

        self.cnn_features = cnn.features

        self.mamba = MambaSequenceClassifier(
            num_classes=num_classes,
            input_dim=self.CNN_OUTPUT_CHANNELS,
            d_model=mamba_d_model,
            n_layers=mamba_layers,
            d_state=mamba_state,
            expand=mamba_expand,
            d_conv=mamba_conv,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Input time series of shape [B, T, F].

            mask:
                Optional validity mask of shape [B, T].
                True = real timestep
                False = padding

        Returns:
            logits of shape [B, num_classes].
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input of shape [B, T, F], "
                f"got {tuple(x.shape)}"
            )

        # [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)

        # ---------------------------------------------------------
        # CNN
        # ---------------------------------------------------------

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(
                    "Expected mask of shape [B, T], "
                    f"got {tuple(mask.shape)}"
                )

            # Adapt the original time-series mask to the CNN input.
            cnn_mask = mask.float().unsqueeze(1)

            cnn_mask = nn.functional.interpolate(
                cnn_mask,
                size=x.shape[-1],
                mode="nearest",
            )

            x = x * cnn_mask

        # [B, F, T] -> [B, 128, T']
        x = self.cnn_features(x)

        # [B, 128, T'] -> [B, T', 128]
        x = x.transpose(1, 2)

        # ---------------------------------------------------------
        # MAMBA
        # ---------------------------------------------------------

        mamba_mask = None

        if mask is not None:

            mamba_mask = nn.functional.interpolate(
                mask.float().unsqueeze(1),
                size=x.shape[1],
                mode="nearest",
            ).squeeze(1).bool()

        logits = self.mamba(
            x,
            mask=mamba_mask,
        )

        return logits
