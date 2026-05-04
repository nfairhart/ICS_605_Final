import sys
import types

# chromadb 1.5.x eagerly imports onnxruntime at package init time, causing a
# 60-120s hang on macOS/Apple Silicon (and unnecessary startup cost on Linux).
# This stub must be applied before any chromadb import anywhere in the app.
_onnx_stub = types.ModuleType("chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2")
_onnx_stub.ONNXMiniLM_L6_V2 = type("ONNXMiniLM_L6_V2", (), {})
sys.modules["chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2"] = _onnx_stub
