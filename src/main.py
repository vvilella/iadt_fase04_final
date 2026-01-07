import argparse
from pathlib import Path
import cv2

from io_video import open_video, get_video_props, make_writer
from frame_loop import process_video_frames
from report import write_report

from context import VideoAnalysisContext
from detectors.face_detector import FaceDetector

from analyzers.emotion_analyzer_openai import EmotionAnalyzerOpenAI
from analyzers.activity_analyzer import ActivityAnalyzer
from analyzers.anomaly_detector import AnomalyDetector
from summary import build_summary



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Caminho do vídeo de entrada (ex: data/sample_video.mp4)")
    p.add_argument("--out_video", default="outputs/annotated.mp4", help="Caminho do vídeo anotado")
    p.add_argument("--out_report", default="outputs/report.json", help="Caminho do relatório final")
    return p.parse_args()


def overlay_basic(frame, frame_idx: int, time_sec: float, fps: float, total_frames: int):
    total_str = str(total_frames) if total_frames > 0 else "?"
    line1 = f"Frame: {frame_idx} / {total_str}"
    line2 = f"Time: {time_sec:.2f}s | FPS: {fps:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3
    thickness = 3
    color = (0, 0, 255)  # vermelho (BGR)

    x, y = 20, 50
    line_gap = 45

    (w1, h1), _ = cv2.getTextSize(line1, font, scale, thickness)
    (w2, h2), _ = cv2.getTextSize(line2, font, scale, thickness)
    w = max(w1, w2)

    pad = 12
    cv2.rectangle(
        frame,
        (x - pad, y - h1 - pad),
        (x + w + pad, y + h2 + pad + line_gap),
        (0, 0, 0),
        -1,
    )

    cv2.putText(frame, line1, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, line2, (x, y + line_gap), font, scale, color, thickness, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    Path("outputs").mkdir(exist_ok=True)

    # --- vídeo input ---
    cap = open_video(args.video)
    props = get_video_props(cap)

    fps = props.get("fps", 0) or 0
    if fps <= 0:
        fps = 30.0

    total_frames = props.get("total_frames", 0) or 0
    width = props["width"]
    height = props["height"]

    writer = make_writer(args.out_video, fps, width, height)

    # --- contexto ---
    context = VideoAnalysisContext()

    # --- detectores/analyzers ---
    face_detector = FaceDetector(min_detection_confidence=0.4, model_selection=1)

    emotion_analyzer = EmotionAnalyzerOpenAI(model="gpt-4o-mini")
    EMOTION_EVERY_N_FRAMES = 30
    last_emotion = None
    last_emotion_conf = None

    activity_analyzer = ActivityAnalyzer()
    ACTIVITY_EVERY_N_FRAMES = 5
    last_activity = None
    last_motion = None

    anomaly_detector = AnomalyDetector(window_size=60, z_thresh=3.0, enable_low=True)
    ANOMALY_COOLDOWN_FRAMES = int(fps * 1.0)      # 1s
    ANOMALY_OVERLAY_FRAMES = int(fps * 2.0)       # 2s
    last_anomaly_frame = -10**9
    anomaly_overlay_until = -1
    anomaly_overlay_text = None

    def on_frame(frame, frame_idx, time_sec):
        nonlocal last_emotion, last_emotion_conf
        nonlocal last_activity, last_motion
        nonlocal last_anomaly_frame, anomaly_overlay_until, anomaly_overlay_text

        # frame "limpo" (sem overlay) para atividade/anomalia
        raw_frame = frame.copy()

        # overlay base (informações)
        frame = overlay_basic(frame, frame_idx, time_sec, fps=fps, total_frames=total_frames)

        # -------- R2: Faces --------
        faces = face_detector.detect(raw_frame)  # detecta no frame limpo
        context.register_faces(len(faces))

        largest = None
        largest_area = 0

        for f in faces:
            x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
            score = f["score"]
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))

            if area > largest_area:
                largest_area = area
                largest = f

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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

        # -------- R3: Emoções (OpenAI) --------
        if largest is not None and (frame_idx % EMOTION_EVERY_N_FRAMES == 0):
            x1, y1, x2, y2 = largest["x1"], largest["y1"], largest["x2"], largest["y2"]
            face_crop = raw_frame[y1:y2, x1:x2].copy()

            emotion, conf = emotion_analyzer.analyze(face_crop)
            if emotion:
                last_emotion = emotion
                last_emotion_conf = conf
                context.register_emotion(emotion)

        if last_emotion:
            txt = f"emotion: {last_emotion}"
            if last_emotion_conf is not None:
                txt += f" ({last_emotion_conf:.2f})"
            cv2.putText(
                frame,
                txt,
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        # -------- R4: Atividades (movimento) --------
        if frame_idx % ACTIVITY_EVERY_N_FRAMES == 0:
            activity, motion = activity_analyzer.analyze(raw_frame)
            last_activity = activity
            last_motion = motion
            context.register_activity(activity)

            # -------- R5: Anomalias (desvio do padrão recente) --------
            anomaly = anomaly_detector.update(motion)
            if anomaly and (frame_idx - last_anomaly_frame) >= ANOMALY_COOLDOWN_FRAMES:
                last_anomaly_frame = frame_idx

                anomaly_overlay_until = frame_idx + ANOMALY_OVERLAY_FRAMES
                anomaly_overlay_text = f"ANOMALY: {anomaly['type']} z={anomaly['z']:.2f}"

                event = {
                    "frame": frame_idx,
                    "time_sec": float(time_sec),
                    "type": anomaly["type"],   # high_motion | low_motion
                    "z": anomaly["z"],
                    "motion": float(motion),
                    "activity": activity,
                }
                context.register_anomaly(event)

        if last_activity:
            txt = f"activity: {last_activity}"
            if last_motion is not None:
                txt += f" | motion={last_motion:.4f}"
            cv2.putText(
                frame,
                txt,
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        # overlay persistente da anomalia
        if anomaly_overlay_text and frame_idx <= anomaly_overlay_until:
            cv2.putText(
                frame,
                anomaly_overlay_text,
                (20, 420),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )

        return frame

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

    summary = build_summary(
    processed_frames=processed_frames,
    fps=fps,
    emotions=dict(context.emotion_counts),
    activities=dict(context.activity_counts),
    anomalies=context.anomalies,
    )

    write_report(
        args.out_report,
        {
            "input_video": args.video,
            "output_video": args.out_video,
            "total_frames_analyzed": processed_frames,
            "frames_with_face_detected": context.frames_with_face,
            "total_face_detections": context.total_face_detections,
            "emotions": dict(context.emotion_counts),
            "activities": dict(context.activity_counts),
            "anomalies_count": len(context.anomalies),
            "anomalies": context.anomalies,
            "summary": summary,
        },
    )

    print("OK!")
    print(f"Frames analisados: {processed_frames}")
    print(f"Vídeo gerado: {args.out_video}")
    print(f"Relatório gerado: {args.out_report}")


if __name__ == "__main__":
    main()
