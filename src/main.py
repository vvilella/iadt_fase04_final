import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

from io_video import open_video, get_video_props, make_writer
from report import write_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Caminho do vídeo de entrada (ex: data/video.mp4)")
    p.add_argument("--out_video", default="outputs/annotated.mp4", help="Caminho do vídeo de saída")
    p.add_argument("--out_report", default="outputs/report.json", help="Caminho do relatório final")
    return p.parse_args()


def main():
    args = parse_args()
    Path("outputs").mkdir(exist_ok=True)

    cap = open_video(args.video)
    props = get_video_props(cap)

    fps = props["fps"]
    if not fps or fps <= 0:
        fps = 30.0  # fallback

    writer = make_writer(args.out_video, fps, props["width"], props["height"])

    processed_frames = 0
    total = props["total_frames"]

    iterator = range(total) if total and total > 0 else None

    if iterator:
        for _ in tqdm(iterator, desc="Processando vídeo"):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            processed_frames += 1
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            processed_frames += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    write_report(
        args.out_report,
        {
            "input_video": args.video,
            "output_video": args.out_video,
            "total_frames_analyzed": processed_frames,
            "anomalies_count": 0,
        },
    )

    print(f"OK! Frames analisados: {processed_frames}")
    print(f"Vídeo gerado: {args.out_video}")
    print(f"Relatório gerado: {args.out_report}")


if __name__ == "__main__":
    main()
