import cv2
import numpy as np


class ActivityAnalyzer:
    """
    Classifica atividade baseada em quantidade de movimento (frame differencing).
    Categorias (simples e defensáveis):
      - still: praticamente sem movimento
      - talking: movimento leve (ex: boca/cabeça)
      - gesturing: movimento moderado/alto (ex: mãos/braços)
    """

    def __init__(self, resize_width: int = 320):
        self.resize_width = resize_width
        self.prev_gray = None

    def _preprocess(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        if w > self.resize_width:
            scale = self.resize_width / float(w)
            bgr_frame = cv2.resize(bgr_frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def analyze(self, bgr_frame):
        """
        Retorna (activity_label, motion_score)
        motion_score ~ percentual (0..1) de pixels "em movimento" no frame reduzido.
        """
        gray = self._preprocess(bgr_frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            return "still", 0.0

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        # threshold para movimento
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # limpa ruído (morfologia)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        motion_pixels = float(cv2.countNonZero(thresh))
        total_pixels = float(thresh.shape[0] * thresh.shape[1])
        motion_score = motion_pixels / total_pixels if total_pixels > 0 else 0.0

        # Heurística de categorias (ajustamos depois com base no vídeo)
        if motion_score < 0.004:
            label = "still"
        elif motion_score < 0.015:
            label = "talking"
        else:
            label = "gesturing"

        return label, float(motion_score)
