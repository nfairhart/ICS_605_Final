"""
model_client.py

Thin abstraction over the scoring backend. Set GEMMA_BACKEND env var to choose:
  lmstudio — fine-tuned Gemma loaded in LM Studio (default; OpenAI-compat endpoint)
  openai   — gpt-4.1-nano via OpenAI API (cloud fallback)

Additional env vars:
  LM_STUDIO_HOST  — LM Studio base URL (default: http://localhost:1234)
  LM_STUDIO_MODEL — model identifier as shown in LM Studio (default: auto-detect loaded model)
  OPENAI_API_KEY  — required for openai backend only
"""

import os
import re
import sys
import types
from pathlib import Path

# Must stub onnxruntime before matching.py imports chromadb
_onnx_stub = types.ModuleType("chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2")
_onnx_stub.ONNXMiniLM_L6_V2 = type("ONNXMiniLM_L6_V2", (), {})
sys.modules["chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"] = _onnx_stub

sys.path.insert(0, str(Path(__file__).parent.parent))

from create_dataset.matching import SYSTEM_PROMPT, USER_TEMPLATE, MatchScore, SCORE_CHARS

# Prompt format used during fine-tuning — must match prepare_finetune_data.py exactly.
# The fine-tuned model was trained with NO system message; using a different format
# causes it to ignore fine-tuning and fall back to base model behaviour.
_FINETUNE_MAX_CHARS = 2500
_FINETUNE_INSTRUCTION = """\
Analyze the match between this resume and job posting.
Return a JSON object with these exact fields:
  rationale            (str, 2-3 sentence overall assessment)
  matching_strengths   (list[str], 2-4 items)
  skill_gaps           (list[str], 2-5 items)
  ats_keywords_missing (list[str], 3-8 items)
  resume_improvements  (list[str], 2-4 items)
  recommended_activities (list[str] or null)
  match_score          (int, 0-100)
  experience_level_fit (str: "under-qualified" | "well-matched" | "over-qualified")

RESUME:
{resume}

JOB POSTING:
{job}"""
from openai import OpenAI


def _parse_match_score(raw: str) -> MatchScore:
    """Parse MatchScore from model output, with JSON extraction fallback."""
    try:
        return MatchScore.model_validate_json(raw)
    except Exception:
        pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        return MatchScore.model_validate_json(m.group())
    raise ValueError(f"Could not parse MatchScore from model output:\n{raw[:500]}")


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


def _lmstudio_client() -> OpenAI:
    host = os.getenv("LM_STUDIO_HOST", "http://localhost:1234")
    return OpenAI(base_url=f"{host}/v1", api_key="lm-studio")


def _lmstudio_model(client: OpenAI) -> str:
    model = os.getenv("LM_STUDIO_MODEL", "")
    if model:
        return model
    # Auto-detect the first loaded model from LM Studio's model list
    try:
        models = client.models.list()
        loaded = [m.id for m in models.data]
        if loaded:
            return loaded[0]
    except Exception:
        pass
    return "local-model"


def _call_openai_compat(client: OpenAI, model: str, messages: list) -> MatchScore:
    """Call any OpenAI-compatible endpoint with structured output, with text fallback."""
    try:
        resp = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=MatchScore,
        )
        result = resp.choices[0].message.parsed
        if result is not None:
            return result
        raw = resp.choices[0].message.content or ""
    except Exception:
        resp = client.chat.completions.create(model=model, messages=messages)
        raw = resp.choices[0].message.content or ""
    return _parse_match_score(raw)


def score_resume_job(resume_text: str, job_text: str) -> MatchScore:
    """Score a (resume, job) pair using the configured backend."""
    backend = os.getenv("GEMMA_BACKEND", "lmstudio").lower()

    resume_text = resume_text[:SCORE_CHARS]
    job_text = job_text[:SCORE_CHARS]
    user_content = USER_TEMPLATE.format(resume_text=resume_text, job_text=job_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if backend == "lmstudio":
        client = _lmstudio_client()
        model = _lmstudio_model(client)
        # Use training-format prompt — no system message, exact instruction template
        finetune_messages = [{"role": "user", "content": _FINETUNE_INSTRUCTION.format(
            resume=resume_text[:_FINETUNE_MAX_CHARS].strip(),
            job=job_text[:_FINETUNE_MAX_CHARS].strip(),
        )}]
        return _call_openai_compat(client, model, finetune_messages)

    elif backend == "openai":
        client = _openai_client()
        resp = client.beta.chat.completions.parse(
            model=os.getenv("OPENAI_SCORING_MODEL", "gpt-4.1-nano"),
            messages=messages,
            response_format=MatchScore,
        )
        return resp.choices[0].message.parsed

    else:
        raise ValueError(f"Unknown GEMMA_BACKEND: {backend!r}. Use 'lmstudio' or 'openai'.")
