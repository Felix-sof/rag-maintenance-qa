"""
Local text embeddings with a multilingual E5 model (sentence-transformers).

Runs on CPU with no API key or quota. multilingual-e5 handles Turkish
questions against Turkish documents (and cross-lingual English <-> Turkish).
The model is downloaded once (~470 MB) on first use and cached under
~/.cache/huggingface.
"""

import os
from typing import Protocol

EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    """multilingual-e5 via sentence-transformers.

    E5 models are trained with "query: " / "passage: " prefixes and retrieve
    noticeably worse without them. Vectors are L2-normalized, so cosine
    similarity == dot product.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        self.name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow import, load lazily

            self._model = SentenceTransformer(self.name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            [f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=16
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._get_model().encode(f"query: {text}", normalize_embeddings=True).tolist()
