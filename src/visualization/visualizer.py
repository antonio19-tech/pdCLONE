import os
import matplotlib.pyplot as plt
from src.visualization.vis_enum import PlotType
from pathlib import Path


class Visualizer:


    def __init__(self):
        self._dispatch = {
            PlotType.XY: self._plot_xy,
            PlotType.TRAJECTORY: self._plot_trajectory,
            PlotType.PRESSURE_OVER_TIME: self._plot_pressure_over_time,
        }


    def plot(self, plot_type: PlotType, sample, **kwargs):
        try:
            plot_fn = self._dispatch[plot_type]
        except KeyError:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        fig = plot_fn(sample, **kwargs)

        filename = kwargs.get("filename", None)
        if filename:
            print(f"Saving plot to {filename}...")

            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)

            with open(path, "rb") as f:
                os.fsync(f.fileno())
        else:
            fig.show()


    def _plot_xy(self, sample, x_feature: str, y_feature: str, z_feature: str = None, **kwargs):
        if x_feature not in sample or y_feature not in sample:
            raise ValueError("X or Y feature not found in sample.")

        fig = plt.figure(figsize=(6, 6), frameon=False, dpi=100)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_facecolor('none')
        fig.add_axes(ax)

        x = sample[x_feature].numpy()
        y = sample[y_feature].numpy()

        if z_feature is not None:
            z = sample[z_feature].numpy()
            ax.scatter(
                x, y,
                c=z,
                cmap=kwargs.get("cmap", "Greys"),
                alpha=kwargs.get("alpha", 1.0),
                s=kwargs.get("markersize", 20),
                marker=kwargs.get("marker", ".")
            )
        else:
            ax.plot(
                x, y,
                marker=kwargs.get("marker", "."),
                linestyle=kwargs.get("linestyle", "-"),
                color=kwargs.get("color", "black"),
                alpha=kwargs.get("alpha", 1.0),
                markersize=kwargs.get("markersize", 5)
            )

        return fig


    def _plot_trajectory(self, sample):
        plt.figure(figsize=(6, 6))
        plt.plot(
            sample["X coordinate"].numpy(),
            sample["Y coordinate"].numpy()
        )
        plt.title("Pen trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


    def _plot_pressure_over_time(self, sample):
        plt.figure(figsize=(6, 6))
        plt.plot(
            sample["time stamp"].numpy(),
            sample["pressure"].numpy()
        )
        plt.title("Pressure over Time")
        plt.xlabel("Time")
        plt.ylabel("Pressure")
        plt.show()