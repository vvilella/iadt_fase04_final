from deepface import DeepFace


class EmotionAnalyzer:
    def __init__(self, enforce_detection: bool = False):
        self.enforce_detection = enforce_detection

    def analyze(self, face_bgr):
        """
        Retorna (dominant_emotion, dominant_score) ou (None, None)
        dominant_score é o score do DeepFace (0..100)
        """
        try:
            result = DeepFace.analyze(
                img_path=face_bgr,
                actions=["emotion"],
                enforce_detection=self.enforce_detection,
            )

            if isinstance(result, list):
                result = result[0]

            dominant = result.get("dominant_emotion")
            emotions = result.get("emotion", {}) or {}

            score = None
            if dominant and dominant in emotions:
                score = float(emotions[dominant])

            return dominant, score
        except Exception:
            return None, None
