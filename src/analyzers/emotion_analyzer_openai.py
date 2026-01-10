import os
import json
import base64
import cv2
from openai import OpenAI


class EmotionAnalyzerOpenAI:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def analyze(self, face_bgr):
        if self.client is None:
            return None, None

        if face_bgr is None or face_bgr.size == 0:
            return None, None

        try:
            ok, buf = cv2.imencode(".jpg", face_bgr)
            if not ok:
                return None, None

            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            prompt = (
                "Classifique a emoção facial predominante em UMA das categorias: "
                "neutral, happy, surprise, sad, fear, disgust. "
                "Responda apenas com JSON no formato: "
                "{\"emotion\":\"<label>\",\"confidence\":0.0}"
            )

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"
                                },
                            },
                        ],
                    }
                ],
            )

            text = (resp.choices[0].message.content or "").strip()

            if not text:
                return None, None

            # Tenta JSON puro primeiro
            try:
                data = json.loads(text)
            except Exception:
                # Fallback: extrai o primeiro objeto JSON {...} encontrado no texto
                import re
                m = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if not m:
                    print("[WARN] OpenAI returned non-JSON:", text[:200])
                    return None, None
                data = json.loads(m.group(0))

            emotion = data.get("emotion")
            conf = data.get("confidence")
            return emotion, conf


        except Exception as e:
            print("[ERROR] EmotionAnalyzerOpenAI:", e)
            return None, None
