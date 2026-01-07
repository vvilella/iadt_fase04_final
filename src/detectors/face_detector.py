import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.6):
        self._mp_face = mp.solutions.face_detection
        self._detector = self._mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, bgr_frame):
        """
        Retorna lista de faces detectadas como dict:
        {x1,y1,x2,y2,score}
        coordenadas em pixels no frame.
        """
        h, w, _ = bgr_frame.shape
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        results = self._detector.process(rgb)
        faces = []

        if not results.detections:
            return faces

        for det in results.detections:
            score = float(det.score[0]) if det.score else 0.0
            box = det.location_data.relative_bounding_box

            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            # clamp
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            faces.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score})

        return faces
