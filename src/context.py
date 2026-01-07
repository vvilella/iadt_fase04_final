class VideoAnalysisContext:
    def __init__(self):
        # R2 - reconhecimento facial
        self.frames_with_face = 0
        self.total_face_detections = 0

    def register_faces(self, num_faces: int):
        if num_faces > 0:
            self.frames_with_face += 1
            self.total_face_detections += num_faces
