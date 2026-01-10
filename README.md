# Tech Challenge – IADT – Fase 4

## Visão Geral

Este projeto implementa uma aplicação de análise de vídeo baseada em visão computacional e IA,
utilizando exclusivamente o vídeo fixo fornecido pelo challenge como entrada, conforme exigido
no enunciado.

O pipeline processa o vídeo frame a frame e gera:
- um vídeo anotado com detecções e eventos
- um relatório estruturado em JSON com métricas consolidadas e um resumo automático

A solução foi construída com foco em clareza arquitetural, decisões técnicas explícitas,
reprodutibilidade e equilíbrio entre processamento local e uso de IA via API.

Todos os requisitos R1 a R6 foram implementados.

---

## Entrada e Saídas

### Entrada
- Vídeo fixo disponibilizado na plataforma do aluno

### Saídas
- outputs/annotated.mp4  
  Vídeo com:
  - detecção facial
  - emoções
  - atividades
  - marcação visual de anomalias

- outputs/report.json  
  Relatório contendo:
  - métricas agregadas
  - lista de anomalias
  - resumo automático do conteúdo do vídeo

---

## Arquitetura Geral

A aplicação utiliza processamento sequencial frame a frame, mantendo um contexto central
para acumular métricas e eventos ao longo do vídeo.

Princípios adotados:
- separação clara de responsabilidades
- módulos independentes
- baixo acoplamento
- comportamento determinístico

Componentes principais:
- FaceDetector (MediaPipe)
- EmotionAnalyzerOpenAI
- ActivityAnalyzer
- AnomalyDetector
- Summary Builder

---

## Bibliotecas e decisões técnicas

### OpenCV
Responsável pela leitura/escrita de vídeo, desenho de overlays e operações básicas de imagem.

### MediaPipe
Utilizado para reconhecimento facial local, garantindo rapidez e estabilidade sem dependência externa.

### OpenAI (Vision)
Utilizado exclusivamente para análise de emoções faciais, executada por amostragem.

A escolha pelo OpenAI ocorreu após tentativa de solução 100% local (DeepFace), que apresentou conflitos
entre TensorFlow, Keras, MediaPipe e Protobuf. A solução híbrida garantiu estabilidade e previsibilidade.

---

## R1 – Vídeo fixo
O sistema opera exclusivamente sobre o vídeo fornecido pelo challenge.

---

## R2 – Reconhecimento Facial
- Detecção local com MediaPipe
- Bounding boxes e score de confiança no vídeo
- Métricas consolidadas no relatório

---

## R3 – Análise de Emoções
- Face detectada localmente
- Emoção inferida via OpenAI Vision por amostragem
- Emoções consideradas:
  - neutral
  - happy
  - surprise
  - sad
  - fear
  - disgust

---

## R4 – Detecção e Categorização de Atividades
Atividade inferida com base no movimento entre frames:
- still
- talking
- gesturing

A análise é feita sempre sobre o frame limpo.

---

## R5 – Detecção de Anomalias
Anomalia definida como desvio estatístico do padrão recente de atividade/movimento.

Implementação:
- janela deslizante
- z-score
- cooldown
- persistência visual no vídeo

---

## R6 – Resumo Automático
Resumo textual local contendo:
- duração do vídeo
- atividades predominantes
- emoções predominantes
- quantidade e exemplos de anomalias

---

## Como executar o projeto

### Criar ambiente virtual
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Instalar dependências
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configurar variável de ambiente
```bash
export OPENAI_API_KEY="sua_chave_aqui"
```

### Executar
```bash
python src/main.py --video data/sample_video.mp4
```

---

## Estrutura do Projeto
```
src/
  analyzers/
  detectors/
  summary.py
  context.py
  main.py
outputs/
  annotated.mp4
  report.json
```

---

## Limitações e extensões futuras
- Emoções 100% locais
- Detecção explícita de mudança de cena
- Agrupamento temporal de anomalias
