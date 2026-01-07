import argparse
from pathlib import Path
import cv2

from io_video import open_video, get_video_props, make_writer
from report import write_report
from frame_loop import process_video_frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Caminho do vídeo de entrada (ex: data/sample_video.mp4)")
    p.add_argument("--out_video", default="outputs/annotated.mp4", help="Caminho do vídeo de saída")
    p.add_argument("--out_report", default="outputs/report.json", help="Caminho do relatório final")
    return p.parse_args()


def overlay_basic(frame, frame_idx: int, time_sec: float, fps: float, total_frames: int | None):
    """
    Overlay simples para provar que o vídeo está sendo processado.
    """
    total_str = str(total_frames) if total_frames and total_frames > 0 else "?"

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

    cap = open_video(args.video)
    props = get_video_props(cap)

    fps = props["fps"]
    if not fps or fps <= 0:
        fps = 30.0  # fallback seguro

    out_video_path = args.out_video
    writer = make_writer(out_video_path, fps, props["width"], props["height"])

    total_frames = props.get("total_frames", 0) or 0

    def on_frame(frame, frame_idx, time_sec):
        return overlay_basic(frame, frame_idx, time_sec, fps=fps, total_frames=total_frames)

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

    write_report(
        args.out_report,
        {
            "input_video": args.video,
            "output_video": args.out_video,
            "total_frames_analyzed": processed_frames,
            "anomalies_count": 0,  # placeholder por enquanto
        },
    )

    print(f"OK! Frames analisados: {processed_frames}")
    print(f"Vídeo gerado: {args.out_video}")
    print(f"Relatório gerado: {args.out_report}")


if __name__ == "__main__":
    main()
