import argparse
from pathlib import Path
import cv2

from io_video import open_video, get_video_props, make_writer
from report import write_report
from frame_loop import process_video_frames
from detectors.face_detector import FaceDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--video",
        required=True,
        help="Caminho do vídeo de entrada (ex: data/sample_video.mp4)",
    )
    p.add_argument(
        "--out_video",
        default="outputs/annotated.mp4",
        help="Caminho do vídeo anotado",
    )
    p.add_argument(
        "--out_report",
        default="outputs/report.json",
        help="Caminho do relatório final",
    )
    return p.parse_args()


def overlay_basic(frame, frame_idx: int, time_sec: float, fps: float, total_frames: int):
    """
    Overlay textual (maior e vermelho) com caixa de fundo.
    """
    total_str = str(total_frames) if total_frames > 0 else "?"

    line1 = f"Frame: {frame_idx} / {total_str}"
    line2 = f"Time: {time_sec:.2f}s | FPS: {fps:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3          # maior
    thickness = 3        # mais “grosso”
    color = (0, 0, 255)  # vermelho em BGR

    x, y = 20, 50
    line_gap = 45

    # Calcula tamanho das duas linhas (para desenhar um fundo)
    (w1, h1), _ = cv2.getTextSize(line1, font, scale, thickness)
    (w2, h2), _ = cv2.getTextSize(line2, font, scale, thickness)
    w = max(w1, w2)
    h = h1 + h2 + line_gap

    # Fundo (preto semi-sólido)
    pad = 12
    cv2.rectangle(
        frame,
        (x - pad, y - h1 - pad),
        (x + w + pad, y + h2 + pad + line_gap),
        (0, 0, 0),
        -1,
    )

    # Texto
    cv2.putText(frame, line1, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, line2, (x, y + line_gap), font, scale, color, thickness, cv2.LINE_AA)

    return frame

    """
    Overlay textual simples (frame / tempo).
    """
    total_str = str(total_frames) if total_frames > 0 else "?"

    cv2.putText(
        frame,
        f"Frame: {frame_idx} / {total_str}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Time: {time_sec:.2f}s | FPS: {fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    args = parse_args()
    Path("outputs").mkdir(exist_ok=True)

    # ---- Abrir vídeo ----
    cap = open_video(args.video)
    props = get_video_props(cap)

    fps = props["fps"]
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = props.get("total_frames", 0) or 0

    writer = make_writer(
        args.out_video,
        fps,
        props["width"],
        props["height"],
    )

    # ---- Detector de face ----
    face_detector = FaceDetector(min_detection_confidence=0.6)

    # ---- Callback de processamento por frame ----
    def on_frame(frame, frame_idx, time_sec):
        # Overlay básico
        frame = overlay_basic(
            frame,
            frame_idx,
            time_sec,
            fps=fps,
            total_frames=total_frames,
        )

        # Detecção de faces
        faces = face_detector.detect(frame)
        for f in faces:
            x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
            score = f["score"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"face {score:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return frame

    # ---- Loop principal ----
    processed_frames = process_video_frames(
        cap=cap,
        writer=writer,
        fps=fps,
        total_frames=total_frames,
        on_frame=on_frame,
    )

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ---- Relatório ----
    write_report(
        args.out_report,
        {
            "input_video": args.video,
            "output_video": args.out_video,
            "total_frames_analyzed": processed_frames,
            "anomalies_count": 0,
        },
    )

    print("OK!")
    print(f"Frames analisados: {processed_frames}")
    print(f"Vídeo gerado: {args.out_video}")
    print(f"Relatório gerado: {args.out_report}")


if __name__ == "__main__":
    main()
