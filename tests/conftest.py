"""Test bootstrap (EMBEDDINGS).

The sandbox cannot write to the default ~/.cache, so every Hugging Face
download (model blobs, locks, temp files) is redirected into the repo's
gitignored .tools/hf-cache before any library import happens.
"""

import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parent.parent / ".tools" / "hf-cache"
os.environ.setdefault("HF_HOME", str(_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE.parent))
