import math
import random
from typing import Callable, Dict, List, Tuple

from quizcraft.utils import log_step

EmbeddingFunc = Callable[[str], List[float]]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_index(chunks: List[Dict[str, str]], embed: EmbeddingFunc) -> List[List[float]]:
    log_step("Build embeddings")
    embeddings: List[List[float]] = []
    for idx, chunk in enumerate(chunks, start=1):
        emb = embed(chunk["text"])
        embeddings.append(emb)
        if idx % 20 == 0:
            print(f"Embedded {idx}/{len(chunks)} chunks...")
    return embeddings


def search_index(
    query: str,
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    embed: EmbeddingFunc,
    top_k: int = 5,
) -> List[Dict[str, str]]:
    query_emb = embed(query)
    scored: List[Tuple[float, Dict[str, str]]] = []
    for chunk, emb in zip(chunks, embeddings):
        scored.append((_cosine_similarity(query_emb, emb), chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def select_chunks(
    chunks: List[Dict[str, str]],
    embeddings: List[List[float]],
    embed: EmbeddingFunc,
    top_k: int,
    seed: int,
    balanced: bool = True,
) -> List[Dict[str, str]]:
    log_step(f"Selector: top {top_k} chunks")
    queries = [
        "definition",
        "theorem",
        "algorithm",
        "conclusion",
        "summary",
        "keypoint",
        "重要",
        "結論",
        "定義",
        "方法",
        "例子",
    ]
    query_embeddings = [embed(q) for q in queries]

    scored: List[Tuple[float, Dict[str, str]]] = []
    for chunk, emb in zip(chunks, embeddings):
        score = max(_cosine_similarity(emb, q_emb) for q_emb in query_embeddings)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    if not balanced:
        return [item[1] for item in scored[:top_k]]

    buckets: Dict[str, int] = {}
    for _, chunk in scored:
        key = chunk.get("section_title") or f"page_{chunk['page']}"
        buckets.setdefault(key, 0)

    max_per_bucket = max(1, top_k // max(1, len(buckets)))

    random.seed(seed)
    selected: List[Dict[str, str]] = []
    remainder: List[Dict[str, str]] = []

    for _, chunk in scored:
        bucket = chunk.get("section_title") or f"page_{chunk['page']}"
        count = buckets.get(bucket, 0)
        if count < max_per_bucket:
            selected.append(chunk)
            buckets[bucket] = count + 1
        else:
            remainder.append(chunk)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        needed = top_k - len(selected)
        selected.extend(remainder[:needed])

    return selected[:top_k]
