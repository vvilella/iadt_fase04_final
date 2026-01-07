# Tech Challenge – IADT – Fase 4

## Visão Geral
Este projeto implementa uma aplicação de análise de vídeo baseada em visão computacional e IA, utilizando um vídeo fixo fornecido pelo challenge como entrada.
O pipeline processa o vídeo frame a frame, gera um vídeo anotado e um relatório estruturado em JSON.

A solução foi construída de forma modular, incremental e defensável, atendendo integralmente aos requisitos R1–R6.

---

## Entrada e Saídas
- Entrada: vídeo fixo disponibilizado na plataforma do aluno
- Saídas:
  - outputs/annotated.mp4 – vídeo com overlays de análise
  - outputs/report.json – relatório estruturado com métricas e resumo automático

---

## Arquitetura Geral
- Processamento frame a frame
- Contexto central para acumular métricas
- Analyzers e detectors desacoplados
- Amostragem para reduzir custo computacional
- Pipeline determinístico e reprodutível

Principais módulos:
- FaceDetector (MediaPipe)
- EmotionAnalyzerOpenAI
- ActivityAnalyzer
- AnomalyDetector
- Summary Builder

---

## R1 – Vídeo fixo
O sistema opera exclusivamente sobre o vídeo fornecido no challenge, conforme especificação do enunciado.

---

## R2 – Reconhecimento Facial
- Implementado com MediaPipe Face Detection
- Detecção local
- Bounding boxes e score de confiança no vídeo
- Métricas no relatório:
  - frames com faces detectadas
  - total de detecções de face

---

## R3 – Análise de Emoções
### Abordagem adotada
- Detecção de face: local
- Classificação de emoção: OpenAI (Vision) por amostragem

### Decisão técnica
Tentativa inicial com DeepFace (100% local) apresentou conflitos entre TensorFlow, Keras, MediaPipe e Protobuf.
Optou-se por uma solução híbrida para garantir estabilidade.

---

## R4 – Detecção e Categorização de Atividades
- Baseada em movimento entre frames (motion score)
- Categorias:
  - still
  - talking
  - gesturing
- Análise realizada sempre sobre frame limpo

---

## R5 – Detecção de Anomalias
### Definição
Anomalia é um desvio estatístico do padrão recente de movimento.

### Implementação
- Janela deslizante
- Z-score
- Cooldown
- Persistência visual no vídeo

---

## R6 – Resumo Automático
Resumo textual local contendo:
- duração do vídeo
- atividades predominantes
- emoções predominantes
- quantidade e exemplos de anomalias

---

## Execução
python src/main.py --video data/sample_video.mp4

---

## Estrutura do Projeto
src/
  analyzers/
  detectors/
  summary.py
  context.py
  main.py

---

## Possíveis Extensões
- Emoções 100% locais
- Detecção explícita de mudança de cena
- Agrupamento temporal de anomalias
