"""
Vector RAG with chunking and optional FAISS backend.

If `faiss` is unavailable the retriever falls back to cosine similarity with
NumPy, so the project still works in constrained environments.
"""
from __future__ import annotations

import csv
import json
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import CFG, DOMAIN_CFG, RAG_DIR, resolve_project_path, logger

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


@dataclass
class RagChunk:
    doc_id: str
    text: str
    source: str
    metadata: dict[str, Any]


_SUPPORTED_SUFFIXES = {".md", ".txt", ".jsonl", ".csv", ".json"}


class EmbeddingBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._fallback_dims = 512
        self.backend_name = "hash"

    def _load_model(self):
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            return None
        try:
            self._model = SentenceTransformer(self.model_name)
            self.backend_name = self.model_name
        except Exception as exc:
            logger.warning("SentenceTransformer unavailable, falling back to hashed embeddings: %s", exc)
            self._model = None
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        if model is not None:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings.astype("float32")
        return self._hashed_encode(texts)

    def _hashed_encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._fallback_dims), dtype="float32")
        for row_idx, text in enumerate(texts):
            for token in text.lower().split():
                vectors[row_idx, hash(token) % self._fallback_dims] += 1.0
            norm = np.linalg.norm(vectors[row_idx])
            if norm > 0:
                vectors[row_idx] /= norm
        return vectors


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _iter_domain_source_files() -> list[Path]:
    files: list[Path] = []
    rag_cfg = DOMAIN_CFG.get("rag", {})
    source_dirs = rag_cfg.get("source_dirs", DOMAIN_CFG.get("rag_source_dirs", ["data/rag"]))
    for source_dir in source_dirs:
        root = resolve_project_path(source_dir)
        if not root.exists():
            continue
        if not _is_within(root, RAG_DIR):
            logger.warning("Ignoring non-RAG retrieval source directory: %s", root)
            continue
        files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES)
    if not files and RAG_DIR.exists():
        files.extend(path for path in RAG_DIR.rglob("*") if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES)
    return sorted(set(files))


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _repair_common_mojibake(text: str) -> str:
    if not text:
        return ""
    markers = ("Ã", "â", "Â", "�")
    if not any(marker in text for marker in markers):
        return text
    try:
        repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text
    if repaired and repaired.count("Ã") + repaired.count("â") < text.count("Ã") + text.count("â"):
        return repaired
    return text


def _format_metadata_context(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    keys = [
        "type",
        "agence",
        "secteur",
        "nb_livraisons_jour",
        "premier_creneau",
        "tous_creneaux",
        "code",
        "nom",
        "marque",
        "geometrie",
        "matiere",
        "photochromique",
        "diametre",
    ]
    items: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        items.append(f"{key}: {rendered}")
    return "\n".join(items)


def _read_text_documents(path: Path) -> Iterable[tuple[str, str, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = _repair_common_mojibake(_read_text_with_fallback(path)).strip()
        if text:
            yield (path.stem, text, {"path": str(path)})
        return

    if suffix == ".jsonl":
        content = _read_text_with_fallback(path)
        for idx, line in enumerate(content.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            text = (
                row.get("text")
                or row.get("content")
                or row.get("body")
                or row.get("description")
                or json.dumps(row, ensure_ascii=False)
            )
            if not text:
                continue
            repaired_text = _repair_common_mojibake(str(text))
            metadata_context = _format_metadata_context(metadata)
            merged_text = f"{repaired_text}\n{metadata_context}".strip() if metadata_context else repaired_text
            merged_metadata = {"path": str(path), "row": idx, **metadata}
            yield (f"{path.stem}_{idx}", merged_text, merged_metadata)
        return

    if suffix == ".csv":
        content = _read_text_with_fallback(path)
        reader = csv.DictReader(io.StringIO(content))
        for idx, row in enumerate(reader):
            text = " | ".join(f"{key}: {value}" for key, value in row.items() if value not in (None, ""))
            if text:
                yield (f"{path.stem}_{idx}", _repair_common_mojibake(text), {"path": str(path), "row": idx})
        return

    if suffix == ".json":
        try:
            payload = json.loads(_read_text_with_fallback(path))
        except json.JSONDecodeError:
            return
        if isinstance(payload, list):
            for idx, row in enumerate(payload):
                text = row.get("text") if isinstance(row, dict) else str(row)
                if text:
                    yield (f"{path.stem}_{idx}", _repair_common_mojibake(str(text)), {"path": str(path), "row": idx})
        elif isinstance(payload, dict):
            text = payload.get("text") or payload.get("content") or json.dumps(payload, ensure_ascii=False)
            if text:
                yield (path.stem, _repair_common_mojibake(str(text)), {"path": str(path)})


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words).strip())
        if start + chunk_size >= len(words):
            break
    return chunks


class VectorRAGRetriever:
    def __init__(self, refresh: bool | None = None) -> None:
        self.refresh = CFG.rag_refresh_on_startup if refresh is None else refresh
        self.embedding_backend = EmbeddingBackend(CFG.embedding_model)
        self.docs: list[RagChunk] = []
        self._matrix: np.ndarray | None = None
        self._index = None
        if self.refresh:
            self.build()

    def build(self) -> None:
        chunks: list[RagChunk] = []
        for path in _iter_domain_source_files():
            for doc_id, text, metadata in _read_text_documents(path):
                split_chunks = _chunk_text(text, CFG.rag_chunk_size, CFG.rag_chunk_overlap)
                for idx, chunk in enumerate(split_chunks):
                    chunks.append(
                        RagChunk(
                            doc_id=f"{doc_id}_chunk_{idx}",
                            text=chunk,
                            source=str(path),
                            metadata=metadata,
                        )
                    )

        self.docs = chunks
        if not chunks:
            self._matrix = np.zeros((0, 1), dtype="float32")
            self._index = None
            return

        embeddings = self.embedding_backend.encode([chunk.text for chunk in chunks])
        self._matrix = embeddings
        if faiss is not None and len(embeddings) > 0:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._index = index
        else:
            self._index = None

        logger.info(
            "RAG ready: %d chunks, backend=%s, faiss=%s",
            len(self.docs),
            self.embedding_backend.backend_name,
            "yes" if self._index is not None else "no",
        )

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        if self._matrix is None:
            self.build()
        if self._matrix is None or len(self.docs) == 0:
            return []

        top_k = top_k or CFG.retrieval_top_k
        query_vec = self.embedding_backend.encode([query])
        if self._index is not None:
            scores, indices = self._index.search(query_vec, min(top_k, len(self.docs)))
            matches = zip(scores[0], indices[0])
        else:
            scores = np.dot(self._matrix, query_vec[0])
            best_indices = np.argsort(scores)[::-1][:top_k]
            matches = ((float(scores[idx]), int(idx)) for idx in best_indices)

        results: list[dict[str, Any]] = []
        for score, idx in matches:
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "score": round(float(score), 4),
                    "source": doc.source,
                    "metadata": doc.metadata,
                }
            )
        return results
