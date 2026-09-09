import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, RecurrencePlot


class TimeSerieToImage:
    def __init__(self, sample: np.ndarray):
        """
        Parameters
        ----------
        sample : np.ndarray
            1D time series array (n_timestamps,)
        """
        self.sample = self._validate_sample(sample)
        self.X = self.sample.reshape(1, -1)  # shape required by pyts

    def _validate_sample(self, sample):
        if not isinstance(sample, np.ndarray):
            raise TypeError("Sample must be a NumPy array.")

        if sample.ndim != 1:
            raise ValueError("Sample must be 1D time series.")

        return sample.astype(float)

    # -------------------------
    # Gramian Angular Field
    # -------------------------
    def to_gaf(self, image_size=32, method="summation"):
        """
        Convert time series to Gramian Angular Field.

        Returns
        -------
        np.ndarray of shape (image_size, image_size)
        """
        gaf = GramianAngularField(
            image_size=image_size,
            method=method
        )
        X_gaf = gaf.fit_transform(self.X)
        return X_gaf[0]

    # -------------------------
    # Recurrence Plot
    # -------------------------
    def to_recurrence_plot(self, dimension=1,
                           threshold='point',
                           percentage=20):
        """
        Convert time series to Recurrence Plot.

        Returns
        -------
        np.ndarray of shape (n_timestamps, n_timestamps)
        """
        rp = RecurrencePlot(
            dimension=dimension,
            threshold=threshold,
            percentage=percentage
        )
        X_rp = rp.fit_transform(self.X)
        return X_rp[0]

    def to_image(self, transform_type: str, **kwargs) -> np.ndarray:
        """
        Dispatches to the selected time-series-to-image transform.
        """
        if transform_type == "gaf":
            return self.to_gaf(
                image_size=kwargs.get("image_size", 32),
                method=kwargs.get("method", "summation"),
            )
        if transform_type == "recurrence_plot":
            return self.to_recurrence_plot(
                dimension=kwargs.get("dimension", 1),
                threshold=kwargs.get("threshold", "point"),
                percentage=kwargs.get("percentage", 20),
            )
        raise ValueError(
            f"Unsupported transform_type '{transform_type}'. "
            "Available: ['gaf', 'recurrence_plot']"
        )

    # -------------------------
    # Visualization
    # -------------------------
    @staticmethod
    def plot(image, title="Image", cmap="viridis"):
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap=cmap, origin="lower")
        plt.title(title)
        plt.colorbar()
        plt.show()

    @staticmethod
    def save(image: np.ndarray, filename: str, cmap: str = "gray") -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, image, cmap=cmap, origin="lower")

        with open(path, "rb") as f:
            os.fsync(f.fileno())
