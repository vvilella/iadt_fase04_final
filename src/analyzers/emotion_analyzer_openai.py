import base64
import json
import cv2
from openai import OpenAI


class EmotionAnalyzerOpenAI:
    """
    Classifica emoção facial em labels simples:
    neutral, happy, sad, angry, fear, surprise, disgust

    Retorna (emotion, confidence_0_1).
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def _bgr_to_data_url(face_bgr) -> str:
        ok, buf = cv2.imencode(".jpg", face_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("Falha ao codificar imagem para JPEG.")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def analyze(self, face_bgr):
        """
        Retorna (emotion, confidence_0_1) ou (None, None).
        """
        try:
            data_url = self._bgr_to_data_url(face_bgr)

            instruction = (
                "Você é um classificador de emoção facial.\n"
                "Retorne APENAS um JSON válido (sem texto extra) no formato:\n"
                '{"emotion":"<label>","confidence":<float_0_1>}\n'
                "Labels permitidos: neutral, happy, sad, angry, fear, surprise, disgust.\n"
                "Se não for possível inferir, use neutral com baixa confiança (ex: 0.2)."
            )

            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": instruction},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
            )

            text = (resp.output_text or "").strip()
            data = json.loads(text)

            emotion = data.get("emotion")
            conf = data.get("confidence")

            if emotion is None:
                return None, None

            conf = float(conf) if conf is not None else None
            # clamp defensivo
            if conf is not None:
                conf = max(0.0, min(1.0, conf))

            return emotion, conf
        except Exception:
            return None, None
