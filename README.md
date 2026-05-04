# ICS 605 Final — Job Application AI

Resume-job matching app using semantic search and LLM scoring.

## Running the app

### LM Studio (default)

Start LM Studio with a model loaded, then:

```bash
streamlit run app/app.py
```

To target a specific host or model:

```bash
LM_STUDIO_HOST=http://localhost:1234 LM_STUDIO_MODEL=your-model-id streamlit run app/app.py
```

### Ollama

```bash
GEMMA_BACKEND=ollama streamlit run app/app.py
```

To use a specific model:

```bash
GEMMA_BACKEND=ollama GEMMA_MODEL=gemma3:9b-instruct-q4_K_M streamlit run app/app.py
```

### OpenAI

`OPENAI_API_KEY` is loaded automatically from `.env` — no need to pass it on the command line.

```bash
GEMMA_BACKEND=openai streamlit run app/app.py
```

To use a specific model:

```bash
GEMMA_BACKEND=openai OPENAI_SCORING_MODEL=gpt-4.1-nano streamlit run app/app.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GEMMA_BACKEND` | `lmstudio` | Backend: `lmstudio`, `ollama`, or `openai` |
| `LM_STUDIO_HOST` | `http://localhost:1234` | LM Studio server URL |
| `LM_STUDIO_MODEL` | *(auto-detect)* | Model ID as shown in LM Studio |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `GEMMA_MODEL` | `gemma3:9b-instruct-q4_K_M` | Ollama model name |
| `OPENAI_API_KEY` | — | Required for OpenAI backend |
| `OPENAI_SCORING_MODEL` | `gpt-4.1-nano` | OpenAI model to use for scoring |
