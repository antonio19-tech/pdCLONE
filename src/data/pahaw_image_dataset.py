from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PaHaWImageDataset(Dataset):
    """
    Reads preprocessed PNG images from a processed folder.
    Each image is labeled as Parkinson (1) or Healthy (0)
    based on the patient's diagnosis from the PaHaWDataset df.

    Filename convention: patient_{pid}_task_{task}.png
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform=None,
    ) -> None:
        self._image_paths = image_paths
        self._labels      = labels
        self._transform   = transform or self.build_transform()

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self._image_paths[idx]).convert("L")
        label = self._labels[idx]
        return self._transform(image), label

    @staticmethod
    def build_transform(
        image_size: int = 64,
        normalization_stats: dict[str, list[float]] | None = None,
    ):
        transform_steps = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if normalization_stats is not None:
            transform_steps.append(
                transforms.Normalize(
                    mean=normalization_stats["mean"],
                    std=normalization_stats["std"],
                )
            )
        return transforms.Compose(transform_steps)

    @staticmethod
    def from_patient_ids(
        patient_ids: list[str],
        processed_dir: str,
        pahaw_dataset,
        disease_column: str = "Disease",
        transform=None,
    ) -> "PaHaWImageDataset":
        """
        Factory: builds an ImageDataset from a list of patient IDs.
        Label is derived from the Disease column: 'PD' -> 1, 'H' -> 0.
        """
        image_paths, labels = [], []
        processed_dir = Path(processed_dir)

        for pid in patient_ids:
            disease = pahaw_dataset.df.loc[int(pid), disease_column]
            label   = 1 if disease == "PD" else 0

            for path in sorted(processed_dir.glob(f"patient_{pid}_task_*.png")):
                image_paths.append(path)
                labels.append(label)

        return PaHaWImageDataset(image_paths, labels, transform)
