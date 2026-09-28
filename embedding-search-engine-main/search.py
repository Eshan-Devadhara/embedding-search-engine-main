"""
search.py - Semantic search engine using precomputed Bedrock embeddings.

Dataset embeddings are generated once at startup (or loaded from cache)
and stored in memory.  Each search only calls Bedrock once - for the
query string.
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from bedrock import generate_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CSV_PATH: str = os.path.join(os.path.dirname(__file__), "subfeatures_dataset.csv")

# Cache stores embeddings as a JSON file alongside the CSV.
# The cache is tagged with the model ID so a model change automatically
# triggers a rebuild.
CACHE_PATH: str = os.path.join(os.path.dirname(__file__), "embeddings_cache.json")

# ---------------------------------------------------------------------------
# In-memory index (populated by init_index / load_index)
# ---------------------------------------------------------------------------

_index: list[dict] | None = None   # list of {name, description, embedding}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_tag(model_id: str) -> str:
    """Return a short tag that identifies the model used for the cache."""
    return model_id.replace("/", "_").replace(":", "_")


def _load_cache(model_id: str) -> list[dict] | None:
    """
    Load the embedding cache from disk.

    Returns None if the cache does not exist or was built with a different
    model.
    """
    if not os.path.isfile(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read embedding cache (%s) - will rebuild.", exc)
        return None

    if data.get("model_tag") != _cache_tag(model_id):
        logger.info(
            "Embedding cache was built with a different model (%s vs %s) - rebuilding.",
            data.get("model_tag"),
            _cache_tag(model_id),
        )
        return None

    records = data.get("records", [])
    if not records:
        return None

    logger.info("Loaded %d embeddings from cache (%s).", len(records), CACHE_PATH)
    return records


def _save_cache(records: list[dict], model_id: str) -> None:
    """Persist embeddings to the local JSON cache."""
    data = {
        "model_tag": _cache_tag(model_id),
        "records": records,
    }
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        logger.info("Saved %d embeddings to cache (%s).", len(records), CACHE_PATH)
    except OSError as exc:
        logger.warning("Could not write embedding cache: %s", exc)


# ---------------------------------------------------------------------------
# Index initialisation
# ---------------------------------------------------------------------------

def init_index(model_id: str, force_rebuild: bool = False) -> None:
    """
    Load the CSV, then either:
      - load precomputed embeddings from the local cache, OR
      - generate embeddings via Bedrock and save them to the cache.

    This function populates the module-level ``_index``.

    Parameters
    ----------
    model_id : str
        Bedrock model ID used for embedding generation.
    force_rebuild : bool
        If True, ignore any existing cache and regenerate all embeddings.
    """
    global _index

    # 1. Load dataset
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        raise RuntimeError(f"Dataset not found at {CSV_PATH!r}.")
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset: {exc}") from exc

    logger.info("Dataset loaded: %d rows.", len(df))

    # 2. Try cache first (unless rebuild requested)
    if not force_rebuild:
        cached = _load_cache(model_id)
        if cached is not None:
            _index = cached
            return

    # 3. Generate embeddings via Bedrock
    logger.info(
        "Generating Bedrock embeddings for %d rows (model=%s). "
        "This may take a moment...",
        len(df),
        model_id,
    )

    records: list[dict] = []
    for i, row in df.iterrows():
        description = str(row.get("Description", "")).strip()
        if not description:
            logger.warning("Row %d has an empty Description - skipping.", i)
            continue
        try:
            embedding = generate_embedding(description)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate embedding for row {i} "
                f"({row.get('Name', '?')!r}): {exc}"
            ) from exc

        records.append(
            {
                "name": str(row.get("Name", "")),
                "description": description,
                "embedding": embedding,
            }
        )
        if (i + 1) % 10 == 0:
            logger.info("  ... %d / %d rows embedded.", i + 1, len(df))

    logger.info("All %d embeddings generated.", len(records))

    # 4. Cache and store in memory
    _save_cache(records, model_id)
    _index = records


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def cosine_similarity(v1: list, v2: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def search_subfeature(description: str, top_k: int = 5) -> list[dict]:
    """
    Search the dataset for the top-k most similar subfeatures.

    Parameters
    ----------
    description : str
        The natural-language search query.
    top_k : int
        Number of results to return (default 5).

    Returns
    -------
    list of dict
        Each dict contains ``name``, ``score`` (float), and ``description``.

    Raises
    ------
    ValueError
        If description is missing or empty.
    RuntimeError
        If the index has not been initialised or Bedrock fails.
    """
    if not description or not description.strip():
        raise ValueError("description query parameter is required and must not be empty.")

    if _index is None:
        raise RuntimeError(
            "Search index is not initialised. Call init_index() on startup."
        )

    # Generate query embedding (single Bedrock call per search)
    query_embedding = generate_embedding(description.strip())

    # Score each record
    results = [
        {
            "name": record["name"],
            "score": cosine_similarity(record["embedding"], query_embedding),
            "description": record["description"],
        }
        for record in _index
    ]

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
