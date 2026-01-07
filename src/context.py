from collections import Counter


class VideoAnalysisContext:
    def __init__(self):
        # R2 - reconhecimento facial
        self.frames_with_face = 0
        self.total_face_detections = 0

        # R3 - emoções (contagem de amostras analisadas)
        self.emotion_counts = Counter()

        # R4 - atividades (contagem de amostras analisadas)
        self.activity_counts = Counter()

        # R5 - anomalias
        self.anomalies = []  # lista de eventos


    def register_faces(self, num_faces: int):
        if num_faces > 0:
            self.frames_with_face += 1
            self.total_face_detections += num_faces

    def register_emotion(self, emotion: str):
        if emotion:
            self.emotion_counts[emotion] += 1

    def register_activity(self, activity: str):
        if activity:
            self.activity_counts[activity] += 1

    def register_anomaly(self, anomaly_event: dict):
        if anomaly_event:
            self.anomalies.append(anomaly_event)

