import torch
import torch.nn as nn
from src.model.mamba_sequence_classifier import MambaSequenceClassifier
from src.model.vanilla_cnn1D import VanillaCNN1D

class CNNMambaClassifier(nn.Module):
    def __init__(
        self,
        segment_input_dim: int,
        segment_embedding_dim: int = 64,
        num_classes: int = 2,
        mamba_d_model: int = 128,
        mamba_layers: int = 2,
        mamba_state: int = 16,
        mamba_expand: int = 2,
        mamba_conv: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.segment_embedding_dim = segment_embedding_dim

        # CNN che trasforma ogni segmento in embedding
        self.segment_encoder = VanillaCNN1D(
            in_channels=segment_input_dim,
            num_classes=2,
            embedding_dim=segment_embedding_dim,
        )

        # Mamba che legge la sequenza di embedding per paziente
        self.mamba = MambaSequenceClassifier(
            input_dim=segment_embedding_dim,
            num_classes=num_classes,
            d_model=mamba_d_model,
            n_layers=mamba_layers,
            d_state=mamba_state,
            expand=mamba_expand,
            d_conv=mamba_conv,
            dropout=dropout,
        )

    def forward(
        self,
        segments: torch.Tensor,  # [B, S, T, F]
        segment_masks: torch.Tensor,  # [B, S, T]
        sequence_mask: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:
        batch_size, n_segments, _, _ = segments.shape

        sequence_mask = sequence_mask.bool()

        # Lunghezza reale di ogni chunk, escluso il padding temporale.
        real_lengths = segment_masks.sum(dim=-1).long()  # [B, S]

        if not sequence_mask.any():
            raise ValueError("The batch does not contain valid segments.")

        embeddings = segments.new_zeros(
            batch_size,
            n_segments,
            self.segment_embedding_dim,
        )

        # Raggruppa i chunk con la stessa lunghezza reale:
        # ogni batch CNN contiene solo valori reali, senza padding interno.
        for length_tensor in torch.unique(real_lengths[sequence_mask]):
            real_length = int(length_tensor.item())

            positions = (
                sequence_mask & (real_lengths == real_length)
            ).nonzero(as_tuple=False)  # [N, 2], colonne: batch_idx, segment_idx

            chunk_batch = torch.stack(
                [
                    segments[b, s, :real_length, :]
                    for b, s in positions.tolist()
                ],
                dim=0,
            )  # [N, real_length, F]

            encoded = self.segment_encoder(
                chunk_batch,
                return_embeddings=True,
            )  # [N, E]

            embeddings[positions[:, 0], positions[:, 1]] = encoded

        return self.mamba(embeddings, mask=sequence_mask)
