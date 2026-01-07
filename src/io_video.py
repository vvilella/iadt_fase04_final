import cv2


def open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Erro ao abrir o vídeo: {video_path}")
    return cap


def get_video_props(cap: cv2.VideoCapture) -> dict:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
    }


def make_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Erro ao criar VideoWriter em: {output_path}")
    return writer
