import numpy as np
import cv2
import matplotlib.pyplot as plt

def segment_to_image(segment: np.ndarray, size: int = 64,
                     col_x: int = 0, col_y: int = 1) -> np.ndarray:
    """
    Converte un segmento 2D (window_size, n_features) in un'immagine mono‑canale.

    - Se n_features >= 2, usa le colonne col_x e col_y come (x, y).
    - Se n_features == 1, disegna il profilo temporale: asse x = indice,
      asse y = valore della feature.
    """
    if segment.ndim == 1:
        segment = segment[:, None]          # rendilo 2D
    n_features = segment.shape[1]

    img = np.zeros((size, size), dtype=np.float32)

    if n_features == 1:
        # Profilo temporale
        t = np.arange(len(segment))
        v = segment[:, 0]
        # Normalizzazione 0‑1
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)
        # Scala a size
        x_int = (t_norm * (size - 1)).astype(np.int32)
        y_int = (v_norm * (size - 1)).astype(np.int32)
    else:
        # Traiettoria usando due feature (x, y)
        x = segment[:, col_x]
        y = segment[:, col_y]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        x_int = (x_norm * (size - 1)).astype(np.int32)
        y_int = (y_norm * (size - 1)).astype(np.int32)

    # Disegno della linea
    for i in range(1, len(x_int)):
        cv2.line(img,
                 (x_int[i-1], y_int[i-1]),
                 (x_int[i],   y_int[i]),
                 1.0, 1, cv2.LINE_AA)

    return img[np.newaxis, :, :]  # (1, H, W)

def show_segment_image(img: np.ndarray, title: str = "Segment", cmap: str = "gray"):
    """Visualizza un'immagine (1, H, W) o (H, W) con matplotlib."""
    img_2d = img.squeeze()          # da (1, H, W) a (H, W)
    plt.imshow(img_2d, cmap=cmap, origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.show()
