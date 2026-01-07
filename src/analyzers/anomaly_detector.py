from collections import deque
import numpy as np


class AnomalyDetector:
    """
    Detecta anomalias a partir do motion_score (0..1), usando janela deslizante.
    Regra:
      - high_motion_anomaly: motion_score muito acima do padrão recente (z-score)
      - low_motion_anomaly: motion_score muito abaixo do padrão recente (opcional)
    """

    def __init__(self, window_size: int = 60, z_thresh: float = 3.0, enable_low: bool = True):
        self.window_size = window_size
        self.z_thresh = z_thresh
        self.enable_low = enable_low
        self.hist = deque(maxlen=window_size)

    def update(self, motion_score: float):
        self.hist.append(float(motion_score))

        # precisa de histórico suficiente
        if len(self.hist) < max(15, self.window_size // 4):
            return None  # sem decisão

        arr = np.array(self.hist, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std()) if float(arr.std()) > 1e-9 else 1e-9

        z = (motion_score - mean) / std

        if z >= self.z_thresh:
            return {"type": "high_motion", "z": float(z), "mean": mean, "std": std}

        if self.enable_low and z <= -self.z_thresh:
            return {"type": "low_motion", "z": float(z), "mean": mean, "std": std}

        return None
