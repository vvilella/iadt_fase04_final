from collections import Counter


def _top_k(counter_like, k=3):
    c = Counter(counter_like)
    return c.most_common(k)


def build_summary(processed_frames: int, fps: float, emotions: dict, activities: dict, anomalies: list):
    duration_sec = (processed_frames / fps) if fps and fps > 0 else None

    top_emotions = _top_k(emotions, 3)
    top_activities = _top_k(activities, 3)

    # pegar até 5 anomalias para citar
    anomaly_points = []
    for a in (anomalies or [])[:5]:
        anomaly_points.append(
            f"{a.get('time_sec', 0):.1f}s ({a.get('type')}, z={a.get('z', 0):.2f})"
        )

    parts = []

    if duration_sec is not None:
        parts.append(f"O vídeo possui aproximadamente {duration_sec:.1f}s ({processed_frames} frames).")
    else:
        parts.append(f"O vídeo possui {processed_frames} frames analisados.")

    if top_activities:
        parts.append(
            "Atividades predominantes (amostragem): "
            + ", ".join([f"{name} ({count})" for name, count in top_activities])
            + "."
        )

    if top_emotions:
        parts.append(
            "Emoções predominantes (amostragem): "
            + ", ".join([f"{name} ({count})" for name, count in top_emotions])
            + "."
        )

    if anomalies:
        parts.append(
            f"Foram detectadas {len(anomalies)} anomalias de movimento (desvio do padrão recente). "
            + ("Exemplos: " + ", ".join(anomaly_points) + "." if anomaly_points else "")
        )
    else:
        parts.append("Não foram detectadas anomalias de movimento relevantes.")

    # texto final
    summary_text = " ".join(parts)

    # também retorna uma versão estruturada
    return {
        "duration_sec": duration_sec,
        "top_emotions": top_emotions,
        "top_activities": top_activities,
        "anomaly_examples": anomaly_points,
        "text": summary_text,
    }
