import torch

from src.model.mamba_sequence_classifier import MambaSequenceClassifier


def test_mamba_sequence_classifier_forward_and_backward():
    model = MambaSequenceClassifier(
        num_classes=2,
        input_dim=3,
        d_model=16,
        n_layers=2,
        d_state=8,
        expand=1,
        d_conv=3,
        dropout=0.1,
    )

    sequences = torch.randn(2, 7, 3)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, False, False],
        ]
    )
    lengths = torch.tensor([7, 4])

    outputs = model(sequences, mask=mask, lengths=lengths)

    assert outputs.shape == (2, 2)

    loss = outputs.sum()
    loss.backward()

    assert model.classifier.weight.grad is not None
