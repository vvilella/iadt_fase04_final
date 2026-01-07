from collections import Counter


class VideoAnalysisContext:
    def __init__(self):
        # R2 - reconhecimento facial
        self.frames_with_face = 0
        self.total_face_detections = 0

        # R3 - emoções (contagem de amostras analisadas)
        self.emotion_counts = Counter()

    def register_faces(self, num_faces: int):
        if num_faces > 0:
            self.frames_with_face += 1
            self.total_face_detections += num_faces

    def register_emotion(self, emotion: str):
        if emotion:
            self.emotion_counts[emotion] += 1
