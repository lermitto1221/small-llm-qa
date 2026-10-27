"""Same-task example selection; query reference answers never enter selection."""
from __future__ import annotations

import random
import time
from pathlib import Path

from .io_utils import canonical_hash, file_hash, write_json


def random_examples(pool, query, n, seed=42):
    candidates = sorted((r for r in pool if r["Task"] == query["Task"]
                         and r["Sample_id"] != query["Sample_id"]), key=lambda r: r["Sample_id"])
    # n-independent shuffle gives nested candidates at n=1,3,5 and order-independent resumes.
    local_seed = int(canonical_hash([seed, query["Task"], query["Sample_id"]])[:16], 16)
    random.Random(local_seed).shuffle(candidates)
    return candidates[:n]


class SimilarityIndex:
    def __init__(self, pool, config, cache_dir):
        import json
        import numpy as np
        from sentence_transformers import SentenceTransformer

        start = time.perf_counter()
        self.pool = sorted(pool, key=lambda r: r["Sample_id"])
        self.config = config
        self.model = SentenceTransformer(config["name"], revision=config["revision"],
                                         device=config.get("device", "cpu"), trust_remote_code=False)
        self.model.max_seq_length = config.get("max_seq_length", 512)
        self._check_lengths([r["Question"] for r in self.pool])
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        identity = {"config": config, "pool_hash": canonical_hash(self.pool)}
        key = canonical_hash(identity)
        array_path, info_path = cache / f"{key}.npy", cache / f"{key}.json"
        self.cache_hit = array_path.exists() and info_path.exists()
        if self.cache_hit:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info["array_sha256"] != file_hash(array_path):
                raise ValueError("Embedding cache checksum mismatch")
            self.vectors = np.load(array_path, allow_pickle=False)
            self.original_index_seconds = info["index_seconds"]
        else:
            self.vectors = self.model.encode([r["Question"] for r in self.pool], batch_size=64,
                                            normalize_embeddings=True, show_progress_bar=True)
            np.save(array_path, self.vectors, allow_pickle=False)
            self.original_index_seconds = time.perf_counter() - start
            write_json(info_path, {**identity, "array_sha256": file_hash(array_path),
                                  "index_seconds": self.original_index_seconds})
        if self.vectors.shape[0] != len(self.pool):
            raise ValueError("Embedding count differs from pool")
        self.preparation_seconds = time.perf_counter() - start

    def _check_lengths(self, questions):
        lengths = [len(x) for x in self.model.tokenizer(questions, truncation=False)["input_ids"]]
        if max(lengths, default=0) > self.model.max_seq_length:
            raise ValueError("Question exceeds MiniLM max_seq_length; inspect before changing encoding policy")

    def select(self, query, n):
        import numpy as np
        start = time.perf_counter()
        self._check_lengths([query["Question"]])
        vector = self.model.encode([query["Question"]], normalize_embeddings=True,
                                   show_progress_bar=False)[0]
        scores = np.dot(self.vectors, vector)
        candidates = [i for i, row in enumerate(self.pool)
                      if row["Task"] == query["Task"] and row["Sample_id"] != query["Sample_id"]]
        order = sorted(candidates, key=lambda i: (-float(scores[i]), self.pool[i]["Sample_id"]))[:n]
        return [self.pool[i] for i in order], time.perf_counter() - start


def choose_common_n(results):
    """results: {n: {'random': macro_score, 'similar': macro_score}}."""
    import math
    ranked = []
    for n, methods in results.items():
        if set(methods) != {"random", "similar"} or any(
            v is None or not math.isfinite(v) or not 0 <= v <= 1 for v in methods.values()
        ):
            raise ValueError("Each n needs finite macro scores for both methods")
        ranked.append((sum(methods.values()) / 2, int(n)))
    if not ranked:
        raise ValueError("No validation results")
    best = max(score for score, _ in ranked)
    return min(n for score, n in ranked if math.isclose(score, best, rel_tol=0, abs_tol=1e-12))
