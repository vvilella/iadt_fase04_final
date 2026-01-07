Tech Challenge – IADT – Fase 4
Análise Inteligente de Vídeo com IA

Este projeto implementa uma aplicação de análise automática de vídeo, com foco em visão computacional e inteligência artificial, conforme os requisitos da Fase 4 da disciplina Inteligência Artificial para Devs (IADT).

A aplicação processa um vídeo fixo fornecido pela plataforma, gera um vídeo anotado e um relatório estruturado em JSON, contendo métricas e inferências realizadas ao longo do processamento.

🎯 Objetivo do Projeto

Construir um pipeline de análise de vídeo capaz de:

Processar um vídeo frame a frame

Detectar rostos

Analisar expressões emocionais

Gerar anotações visuais no vídeo

Produzir um relatório final consolidado

A arquitetura foi pensada de forma modular, permitindo a evolução incremental do projeto.

🧱 Arquitetura Geral

O pipeline atual é composto por:

Leitura e escrita de vídeo (OpenCV)

Loop de processamento de frames com callback (on_frame)

Contexto de análise para acumular métricas globais

Módulos independentes para cada tipo de análise (faces, emoções, etc.)

Relatório final em JSON

video -> frames -> análises -> vídeo anotado + report.json

📁 Estrutura do Projeto
iadt_fase04_final/
├── src/
│   ├── main.py                 # Pipeline principal
│   ├── frame_loop.py           # Loop genérico de frames
│   ├── io_video.py             # Abertura e escrita de vídeo
│   ├── report.py               # Geração do relatório final
│   ├── context.py              # Contexto global de métricas
│   ├── detectors/
│   │   └── face_detector.py    # Detecção facial (MediaPipe)
│   └── analyzers/
│       └── emotion_analyzer_openai.py  # Emoções via OpenAI
├── data/
│   └── sample_video.mp4        # Vídeo de entrada (ignorado no git)
├── outputs/
│   ├── annotated.mp4           # Vídeo anotado
│   └── report.json             # Relatório final
├── requirements.txt
└── README.md

✅ Funcionalidades Implementadas até o Momento
🔹 R1 – Vídeo de Entrada Fixo

O vídeo é fornecido como entrada fixa conforme o enunciado.

O pipeline assume um único vídeo de análise por execução.

🔹 R2 – Reconhecimento Facial

Implementado com MediaPipe Face Detection (execução local).

Para cada frame:

Detecta rostos

Desenha bounding boxes no vídeo

Métricas coletadas:

Total de frames analisados

Frames com pelo menos um rosto

Total de detecções faciais

Essas informações são incluídas no relatório final (report.json).

🔹 R3 – Análise de Emoções (Abordagem Híbrida)
Abordagem Atual (Implementada)

Detecção facial local (MediaPipe)

Classificação de emoções via OpenAI Vision API

Emoção analisada por amostragem temporal (a cada N frames)

A última emoção válida é mantida como cache para exibição contínua no vídeo

Emoções consideradas:

neutral

happy

sad

angry

fear

surprise

disgust

O resultado é:

Exibido no vídeo anotado

Agregado no relatório final como contagem de ocorrências

🧪 Tentativa Inicial: Emoções 100% Locais (Não Utilizada)

Inicialmente, foi avaliada uma abordagem totalmente local para análise de emoções utilizando a biblioteca DeepFace (baseada em TensorFlow/Keras).

No entanto, durante a integração com o MediaPipe, surgiram conflitos de dependências, especialmente envolvendo:

TensorFlow / Keras

Versões incompatíveis de protobuf

Esses conflitos inviabilizaram a execução estável de ambos os componentes no mesmo ambiente Python.

Decisão Arquitetural

Optou-se por uma abordagem híbrida, que:

Mantém a detecção facial local

Remove dependências pesadas (TensorFlow/Keras)

Garante estabilidade do ambiente

Preserva a modularidade do pipeline

Essa decisão foi tomada visando robustez, clareza e prazo, sem comprometer os objetivos do trabalho.

📊 Exemplo de Relatório Gerado
{
  "total_frames_analyzed": 3326,
  "frames_with_face_detected": 2033,
  "total_face_detections": 2449,
  "emotions": {
    "neutral": 19,
    "happy": 7,
    "surprise": 5,
    "sad": 2,
    "disgust": 1,
    "fear": 1
  },
  "anomalies_count": 0
}

⚙️ Requisitos e Execução
Dependências principais

Python 3.11

OpenCV

MediaPipe

OpenAI SDK

Execução
python src/main.py --video data/sample_video.mp4


⚠️ Para R3, é necessário definir a variável de ambiente OPENAI_API_KEY.

🧩 Limitações Conhecidas e Trabalhos Futuros

A análise de emoções depende atualmente de uma API externa (OpenAI).

Como melhoria futura (bônus), o módulo de emoções pode ser substituído por um classificador local baseado em CNNs pré-treinadas (ex: FER2013 ou AffectNet).

A arquitetura foi projetada para permitir essa substituição sem impacto nos demais módulos.

🚧 Próximos Passos Planejados

R4 – Detecção e categorização de atividades

R5 – Detecção de anomalias

R6 – Geração de resumo automático do vídeo

Essas funcionalidades serão implementadas de forma incremental, com commits separados.

✍️ Autor

Victor Nardi Vilella
Pós-Graduação – IA para Devs (FIAP)