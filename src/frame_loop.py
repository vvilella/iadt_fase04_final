from typing import Callable, Optional
import cv2
from tqdm import tqdm


def process_video_frames(
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    fps: float,
    total_frames: Optional[int],
    on_frame: Callable[[cv2.Mat, int, float], cv2.Mat],
) -> int:
    """
    Percorre o vídeo frame a frame, aplica um processamento
    e escreve no writer.

    on_frame(frame, frame_index, time_sec) -> frame processado
    """
    processed = 0

    iterator = range(total_frames) if total_frames and total_frames > 0 else None

    if iterator:
        for _ in tqdm(iterator, desc="Processando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = processed + 1
            time_sec = frame_idx / fps if fps else 0.0

            frame = on_frame(frame, frame_idx, time_sec)
            writer.write(frame)
            processed += 1
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = processed + 1
            time_sec = frame_idx / fps if fps else 0.0

            frame = on_frame(frame, frame_idx, time_sec)
            writer.write(frame)
            processed += 1

    return processed
