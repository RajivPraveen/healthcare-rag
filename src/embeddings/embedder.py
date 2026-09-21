"""Module 4 - embedding generation.

Wraps a Sentence-Transformers bi-encoder with the two details that matter for
retrieval quality and for iterating quickly:

1. **Asymmetric prompts.** BGE/E5-family models are trained with an instruction
   prefix on the *query* side only. Embedding a question the same way as a
   passage costs several points of recall, so queries and passages take
   different code paths.
2. **On-disk cache.** Re-embedding 3.5k chunks on every experiment run wastes
   minutes. Vectors are memo-ised by (model, text) hash so a chunking sweep
   only pays for genuinely new text.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

from src.common.config import get_settings
from src.common.logging import get_logger

log = get_logger(__name__)

# Instruction prefixes applied to queries only, keyed by model family.
_QUERY_PROMPTS: dict[str, str] = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
    "gte": "",
}


def _query_prefix(model_name: str) -> str:
    lowered = model_name.lower()
    for family, prompt in _QUERY_PROMPTS.items():
        if family in lowered:
            return prompt
    return ""


def _passage_prefix(model_name: str) -> str:
    return "passage: " if "e5" in model_name.lower() else ""


class Embedder:
    """Lazy-loading, cached sentence embedder."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_dir: Path | None = None,
        batch_size: int = 64,
        use_cache: bool = True,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.resolve_device()
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache_dir = cache_dir or (settings.data_dir / ".cache" / "embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._lock = threading.Lock()
        self._mem: dict[str, np.ndarray] = {}
        self._cache_path = self.cache_dir / f"{self.model_name.replace('/', '__')}.npz"
        self._load_cache()

    # -- model ---------------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("loading embedding model %s on %s", self.model_name, self.device)
                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    # -- cache ---------------------------------------------------------------

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not (self.use_cache and self._cache_path.exists()):
            return
        try:
            with np.load(self._cache_path, allow_pickle=False) as data:
                keys = data["keys"]
                vecs = data["vectors"]
            self._mem = {str(k): vecs[i] for i, k in enumerate(keys)}
            log.info("embedding cache: %d vectors loaded", len(self._mem))
        except Exception as exc:
            log.warning("could not read embedding cache (%s); starting fresh", exc)
            self._mem = {}

    def save_cache(self) -> None:
        if not self.use_cache or not self._mem:
            return
        keys = np.array(list(self._mem.keys()))
        vectors = np.vstack(list(self._mem.values()))
        np.savez_compressed(self._cache_path, keys=keys, vectors=vectors)
        log.info("embedding cache: %d vectors saved", len(keys))

    # -- encoding ------------------------------------------------------------

    def _encode_raw(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity becomes a dot product
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Embed documents/chunks, using the cache where possible."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        prefix = _passage_prefix(self.model_name)
        prepared = [prefix + t for t in texts]
        keys = [self._key(t) for t in prepared]

        missing_idx = [i for i, k in enumerate(keys) if k not in self._mem]
        if missing_idx:
            log.info(
                "embedding %d passages (%d served from cache)",
                len(missing_idx),
                len(texts) - len(missing_idx),
            )
            fresh = self._encode_raw([prepared[i] for i in missing_idx], show_progress)
            for slot, i in enumerate(missing_idx):
                self._mem[keys[i]] = fresh[slot]

        return np.vstack([self._mem[k] for k in keys]).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.dimension), dtype=np.float32)
        prefix = _query_prefix(self.model_name)
        return self._encode_raw([prefix + q for q in queries], show_progress=False)


_DEFAULT: Embedder | None = None


def get_embedder(model_name: str | None = None) -> Embedder:
    """Process-wide default embedder (avoids reloading the model per request)."""
    global _DEFAULT
    if _DEFAULT is None or (model_name and model_name != _DEFAULT.model_name):
        _DEFAULT = Embedder(model_name=model_name)
    return _DEFAULT
