# ICS 605 Final — Job Application AI

Resume-job matching app using semantic search and LLM scoring.

## Setup

### API key

Create a `.env` file at the project root with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

The key is used for embeddings (`text-embedding-3-small`) and scoring pairs with `gpt-4.1-nano` during dataset generation. The app also reads it when the OpenAI backend is selected.

---

## Building training data

The dataset pipeline runs in five steps. All scripts live in `create_dataset/` and read `OPENAI_API_KEY` from `.env`.

You will also need two external datasets:
- **Resume dataset** — place the Kaggle resume CSV at `resume-dataset/Resume/Resume.csv`
- **LinkedIn job postings** — place the CSV at `linkedin-job-postings/postings.csv`

**Step 1 — Extract resume text**

```bash
python create_dataset/convert_pdfs.py
```

Reads `Resume.csv` and writes `resume_texts.json`. Pass `--from-pdf` to extract directly from PDFs instead.

**Step 2 — Extract job posting text**

```bash
python create_dataset/convert_jobs.py
# or limit to a smaller sample:
python create_dataset/convert_jobs.py --limit 10000
```

Reads `postings.csv` and writes `job_texts.json`.

**Step 3 — Build vector embeddings**

```bash
python create_dataset/embedding.py                 # embed resumes
python create_dataset/embedding.py --dataset jobs  # embed job postings
```

Calls the OpenAI embedding API and stores vectors in a local ChromaDB at `chroma_db/`.

**Step 4 — Score resume–job pairs**

```bash
python create_dataset/matching.py \
    --sample-per-category 20 \
    --top-k 2 \
    --add-random-jobs 2 \
    --workers 10
```

For each resume, retrieves the top-k semantically similar jobs from ChromaDB, scores each pair with `gpt-4.1-nano`, and appends results to `sample_matches.jsonl`. The script is resumable — re-running skips already-scored pairs.

**Step 5 — Format for fine-tuning**

```bash
python create_dataset/prepare_finetune_data.py
```

Reads `sample_matches.jsonl` and writes `finetune_data/train.jsonl` and `finetune_data/val.jsonl` in Gemma 4 chat format (9,000 train / remainder val, shuffled with seed 42).

---

## Training on Google Colab

The notebook `training/train_gemma4_e2b.ipynb` fine-tunes Gemma 4 E2B with LoRA using [Unsloth](https://github.com/unslothai/unsloth). It requires an **A100 GPU** (Colab Pro/Pro+).

**Before running:**

1. Upload `finetune_data/train.jsonl` and `finetune_data/val.jsonl` to Google Drive at `MyDrive/finetune_data/`.
2. Open the notebook in Colab and set the runtime to **A100 GPU**.

**Notebook cells:**

| Cell | What it does |
|---|---|
| 1 | Installs Unsloth and dependencies |
| 2 | Mounts Google Drive, sets `DATA_DIR` |
| 3 | Loads Gemma 4 E2B and applies LoRA (r=8, q_proj + v_proj) |
| 4 | Loads train/val datasets from Drive |
| 5 | Sanity-checks sequence lengths |
| 6 | Trains with SFTTrainer (3 epochs, early stopping) |
| 6b | Saves the training loss chart to Drive |
| 7 | Saves the LoRA adapter as a zip to Drive |
| 8 | Converts to GGUF (Q4\_K\_M) and copies to Drive |

After training, the fine-tuned GGUF file in Drive can be loaded directly into LM Studio.

---

## Models

Fine-tuned models are available on Google Drive:
[Download models](https://drive.google.com/drive/folders/1BBtvL7N8qFJH_3LToX_7b03w1jfnuRRq?usp=sharing)

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
