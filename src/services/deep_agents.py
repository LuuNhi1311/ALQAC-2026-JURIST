from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import warnings

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

warnings.filterwarnings("ignore", message="Pydantic serializer warnings:")
warnings.filterwarnings("ignore", message=".*get_sentence_embedding_dimension.*")
warnings.filterwarnings("ignore", message=".*Failed to obtain server version.*")

LLM_PROVIDER_AZURE = "azure"
LLM_PROVIDER_VLLM = "vllm"
LLM_PROVIDERS = (LLM_PROVIDER_AZURE, LLM_PROVIDER_VLLM)

_DEFAULT_MIN_INTERVAL = {LLM_PROVIDER_AZURE: 10.0, LLM_PROVIDER_VLLM: 0.0}


class _Tracer:
    """Per-case debug recorder. Off by default; enabled when a trace path is set.

    Components (law retrieval, case API, the outcome prompt) push into the current
    case's record only while enabled, so production runs pay nothing.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.law_calls: List[Dict[str, Any]] = []
        self.case_calls: List[Dict[str, Any]] = []
        self.classification_prompt: Optional[str] = None

    def reset(self) -> None:
        self.law_calls = []
        self.case_calls = []
        self.classification_prompt = None


TRACE = _Tracer()


@dataclass(frozen=True)
class Settings:
    # Runtime (shared: embedding models + outcome classifier)
    device: str

    # Data / IO (Application)
    corpus_path: Path
    input_path: Path
    output_path: Path

    # Vector store: Qdrant (QdrantClientFactory / QdrantLawRepository)
    collection_name: str
    qdrant_url: Optional[str]
    qdrant_api_key: Optional[str]
    qdrant_local_path: Path

    # Embedding models (EmbedderFactory)
    dense_model_name: str
    sparse_model_name: str

    # LLM (LanguageModelFactory)
    llm_provider: str
    llm_model_name: str
    llm_temperature: float
    llm_max_retries: int
    llm_min_interval: float
    llm_retry_delay: float

    # Self-hosted vLLM / OpenAI-compatible endpoint (OpenAICompatLanguageModel)
    vllm_api_base: str
    vllm_model_name: str
    vllm_api_key: str

    # Case content API (CaseContentClient)
    case_api_url: str
    case_api_token: str
    api_min_interval: float
    api_rate_limit_delay: float
    api_max_rate_limit_retries: int
    api_max_retries: int

    # Law retrieval agent (LawRetrievalAgent)
    law_search_limit: int
    max_retrieval_iterations: int
    max_law_evidence: int

    # Case evidence collector (CaseEvidenceCollector)
    max_case_queries: int

    # Outcome classifier (ClassifierOutcomePredictor)
    outcome_model_id: str
    outcome_max_length: int

    # Outcome prediction strategy (LlmOutcomePredictor / ClassifierOutcomePredictor)
    predictor_kind: str
    outcome_samples: int
    outcome_temperature: float

    # Law evidence selection (LawEvidenceSelector)
    law_select: bool
    law_evidence_target: int

    use_reranker: bool
    rerank_model_name: str
    law_rerank_topk: int
    law_rerank_min_score: float

    # Debug trace (per-case JSON dump of api/law calls + classification prompt)
    trace_output: str

    @staticmethod
    def from_environment(overrides: Optional[Dict[str, Any]] = None) -> "Settings":
        source: Dict[str, Any] = dict(overrides or {})
        base = Path(__file__).resolve().parents[2]
        host = os.environ.get("LOCAL_QDRANT_URL") or "http://localhost:6333"
        device = source.get("device") or ("cuda" if _cuda_available() else "cpu")
        provider = (
            str(source.get("llm_provider", os.environ.get("LLM_PROVIDER", LLM_PROVIDER_AZURE)))
            .strip()
            .lower()
        )
        if provider not in LLM_PROVIDERS:
            raise ValueError(
                f"llm_provider={provider!r} is not supported. Choose one of: {', '.join(LLM_PROVIDERS)}"
            )
        return Settings(
            # Runtime
            device=str(device),
            # Data / IO
            corpus_path=Path(source.get("corpus_path", base / "data" / "corpus_law_pub.json")),
            input_path=Path(source.get("input_path", base / "data" / "ALQAC2026_public_test.json")),
            output_path=Path(source.get("output_path", base / "submission_deep_agents.json")),
            # Vector store: Qdrant
            collection_name=str(source.get("collection_name", "alqac_law_corpus")),
            qdrant_url=source.get("qdrant_url", host),
            qdrant_api_key=source.get("qdrant_api_key", os.environ.get("LOCAL_QDRANT_API_KEY")),
            qdrant_local_path=Path(
                source.get(
                    "qdrant_local_path",
                    os.environ.get("QDRANT_LOCAL_PATH", base / "qdrant_storage_local"),
                )
            ),
            # Embedding models
            dense_model_name=str(source.get("dense_model_name", "leonpham1208/alqac_halong_embedding")),
            sparse_model_name=str(source.get("sparse_model_name", "Qdrant/bm25")),
            # LLM
            llm_provider=provider,
            llm_model_name=str(source.get("llm_model_name", "azure/gpt-4o")),
            llm_temperature=float(source.get("llm_temperature", 0.1)),
            llm_max_retries=int(source.get("llm_max_retries", 3)),
            llm_min_interval=float(
                source.get(
                    "llm_min_interval",
                    os.environ.get("LLM_MIN_INTERVAL", _DEFAULT_MIN_INTERVAL[provider]),
                )
            ),
            llm_retry_delay=float(source.get("llm_retry_delay", 5.0)),
            # Self-hosted vLLM / OpenAI-compatible endpoint
            vllm_api_base=str(
                source.get("vllm_api_base", os.environ.get("VLLM_API_BASE", "http://localhost:8001/v1"))
            ),
            vllm_model_name=str(
                source.get("vllm_model_name", os.environ.get("VLLM_MODEL_NAME", "vietnamese-law"))
            ),
            vllm_api_key=str(source.get("vllm_api_key", os.environ.get("VLLM_API_KEY", "EMPTY"))),
            # Case content API
            case_api_url=str(source.get("case_api_url", "https://alqac-api.ngrok.pro/retrieve")),
            case_api_token=str(source.get("case_api_token", os.environ.get("ALQAC_TOKEN", ""))),
            api_min_interval=float(
                source.get("api_min_interval", os.environ.get("CASE_API_MIN_INTERVAL", 6.0))
            ),
            api_rate_limit_delay=float(source.get("api_rate_limit_delay", 7.0)),
            api_max_rate_limit_retries=int(source.get("api_max_rate_limit_retries", 10)),
            api_max_retries=int(source.get("api_max_retries", 3)),
            # Law retrieval agent
            law_search_limit=int(source.get("law_search_limit", 10)),
            max_retrieval_iterations=int(source.get("max_retrieval_iterations", 3)),
            max_law_evidence=int(source.get("max_law_evidence", 12)),
            # Case evidence collector
            max_case_queries=int(source.get("max_case_queries", 16)),
            # Outcome classifier
            outcome_model_id=str(
                source.get("outcome_model_id", "leonpham1208/alqac_legal_outcome_cls")
            ),
            outcome_max_length=int(source.get("outcome_max_length", 512)),
            # Outcome prediction strategy
            predictor_kind=str(source.get("predictor_kind", "llm")).strip().lower(),
            outcome_samples=int(source.get("outcome_samples", 3)),
            outcome_temperature=float(source.get("outcome_temperature", 0.4)),
            # Law evidence selection
            law_select=_as_bool(source.get("law_select", True)),
            law_evidence_target=int(source.get("law_evidence_target", 8)),
            trace_output=str(source.get("trace_output", os.environ.get("TRACE_OUTPUT", ""))),
            use_reranker=_as_bool(source.get("use_reranker", True)),
            rerank_model_name=str(
                source.get("rerank_model_name", "AITeamVN/Vietnamese_Reranker")
            ),
            law_rerank_topk=int(source.get("law_rerank_topk", 8)),
            law_rerank_min_score=float(source.get("law_rerank_min_score", 0.0)),
        )


_PROGRESS_STYLE = {
    "index": ("📚", "green"),
    "inference": ("⚖️", "cyan"),
}


def _progress(total: int, desc: str, unit: str) -> Any:
    """Progress bar tuned for both an interactive terminal and a redirected log.

    When stderr is not a tty the refresh interval is widened so a long run adds a
    handful of lines to the log file instead of thousands of carriage returns.
    """
    interactive = sys.stderr.isatty()
    emoji, colour = _PROGRESS_STYLE.get(desc, ("✨", "magenta"))
    return tqdm(
        total=total,
        desc=f"{emoji} {desc}",
        unit=unit,
        colour=colour,
        dynamic_ncols=True,
        mininterval=0.3 if interactive else 30.0,
        bar_format=(
            "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] 🚀"
        ),
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class LawArticle:
    law_id: str
    aid: int
    order: int
    text: str

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"alqac-law:{self.law_id}:{self.aid}"))


@dataclass(frozen=True)
class LawReference:
    aid: int
    law_id: str
    text: str = ""
    score: float = 0.0

    def as_evidence(self) -> Dict[str, Any]:
        return {"law_id": self.law_id, "aid": self.aid}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True)
class LegalCase:
    case_id: str
    case_query: str
    a_role: str
    b_role: str
    a_description: str
    b_description: str
    case_type: str
    court: str
    full_text: str = ""

    def context(self) -> str:
        return (
            f"Loại vụ án: {self.case_type}\n"
            f"Tòa án: {self.court}\n"
            f"Bên A ({self.a_role}): {self.a_description}\n"
            f"Bên B ({self.b_role}): {self.b_description}\n"
            f"Nội dung tranh chấp: {self.case_query}"
        )

    @staticmethod
    def from_record(record: Dict[str, Any]) -> "LegalCase":
        full_text = " ".join(
            str(record.get(field) or "").strip()
            for field in ("case_fact", "court_reasoning", "court_verdict")
        ).strip()
        return LegalCase(
            case_id=str(record.get("case_id", "")),
            case_query=str(record.get("case_query", "")).strip(),
            a_role=str(record.get("A_role", "Nguyên đơn")).strip(),
            b_role=str(record.get("B_role", "Bị đơn")).strip(),
            a_description=str(record.get("A_description", "")).strip(),
            b_description=str(record.get("B_description", "")).strip(),
            case_type=str(record.get("case_type", "")).strip(),
            court=str(record.get("court", "")).strip(),
            full_text=full_text,
        )


@dataclass(frozen=True)
class CaseQueryDecomposition:
    keywords: List[str]
    sub_queries: List[str]


@dataclass(frozen=True)
class RetrievalAssessment:
    sufficient: bool
    missing_information: List[str]
    additional_keywords: List[str]


@dataclass(frozen=True)
class OutcomeReasoning:
    prediction: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class SubmissionEntry:
    case_id: str
    case_evidence: List[str]
    law_evidence: List[Dict[str, Any]]
    prediction: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prediction": self.prediction,
            "case_evidence": list(self.case_evidence),
            "law_evidence": list(self.law_evidence),
        }


VALID_LABELS: Tuple[str, ...] = ("A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN")
DEFAULT_PREDICTION: str = "PARTIAL_A_WIN"

PENALTY_SAFE_MIN_SEGMENTS: int = 5
RETRIEVAL_DRY_STREAK: int = 3
SPREAD_QUERY_WINDOW: int = 10
SPREAD_QUERY_COUNT: int = 40

JUDGMENT_PROBE_QUERIES: Tuple[str, ...] = (
    "Quyết định của Tòa án tuyên xử",
    "Chấp nhận một phần yêu cầu khởi kiện của nguyên đơn",
    "Không chấp nhận một phần yêu cầu khởi kiện",
    "Bác một phần yêu cầu của nguyên đơn",
    "Chấp nhận toàn bộ yêu cầu khởi kiện của nguyên đơn",
    "Không chấp nhận toàn bộ yêu cầu khởi kiện",
    "Căn cứ áp dụng các Điều khoản của Bộ luật Dân sự",
    "án phí dân sự sơ thẩm Nghị quyết 326 lệ phí Tòa án",
    "Buộc bị đơn phải bồi thường cho nguyên đơn số tiền",
    "Nhận định của Hội đồng xét xử về nội dung tranh chấp",
    "lời khai trình bày của nguyên đơn",
    "ý kiến trình bày của bị đơn",
    "chứng cứ tài liệu có trong hồ sơ vụ án",
    "kết quả thẩm định giá tài sản",
)


class CorpusLoader(Protocol):
    def load(self) -> List[LawArticle]: ...


class JsonLawCorpusLoader:
    def __init__(self, corpus_path: Path) -> None:
        self._corpus_path = corpus_path

    def load(self) -> List[LawArticle]:
        with self._corpus_path.open("r", encoding="utf-8") as handle:
            documents = json.load(handle)
        articles: List[LawArticle] = []
        for document in documents:
            law_id = str(document.get("law_id", "")).strip()
            for order, article in enumerate(document.get("content", [])):
                text = str(article.get("content_Article", "")).strip()
                if not text:
                    continue
                articles.append(
                    LawArticle(law_id=law_id, aid=int(article["aid"]), order=order, text=text)
                )
        return articles


@runtime_checkable
class DenseEmbedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def encode_query(self, text: str) -> List[float]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> List[Tuple[List[int], List[float]]]: ...

    def encode_query(self, text: str) -> Tuple[List[int], List[float]]: ...


class HalongDenseEmbedder:
    def __init__(self, model_name: str, device: str, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def encode_query(self, text: str) -> List[float]:
        return self.encode_documents([text])[0]


class Reranker:
    def __init__(self, model_name: str, device: str, max_length: int = 1024) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._model: Any = None
        self._failed = False

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from sentence_transformers import CrossEncoder

            device = self._device if self._device.startswith("cuda") else "cpu"
            self._model = CrossEncoder(
                self._model_name, device=device, max_length=self._max_length
            )
            print(f"🏆 [rerank] loaded {self._model_name} on {device}", flush=True)
            return True
        except Exception as error:
            print(f"🏆 [rerank] unavailable ({error}); using identity order", flush=True)
            self._failed = True
            return False

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts or not self._ensure():
            return [0.0] * len(texts)
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]

    def order(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        keep_min: int = 2,
    ) -> List[RetrievedChunk]:
        items = list(chunks)
        if len(items) <= 1 or not self._ensure():
            return items[:top_k] if top_k else items
        scores = self.score(query, [c.text for c in items])
        paired = sorted(zip(scores, items), key=lambda p: p[0], reverse=True)
        if min_score > 0.0:
            kept = [(s, c) for s, c in paired if s >= min_score]
            paired = kept if len(kept) >= keep_min else paired[:keep_min]
        ranked = [c for _, c in paired]
        return ranked[:top_k] if top_k else ranked


class Bm25SparseEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model_name)

    def encode_documents(self, texts: Sequence[str]) -> List[Tuple[List[int], List[float]]]:
        results: List[Tuple[List[int], List[float]]] = []
        for embedding in self._model.embed(list(texts)):
            results.append((embedding.indices.tolist(), embedding.values.tolist()))
        return results

    def encode_query(self, text: str) -> Tuple[List[int], List[float]]:
        embedding = next(iter(self._model.query_embed(text)))
        return embedding.indices.tolist(), embedding.values.tolist()


# Process-wide cache of loaded models so an embedder/reranker is built once and
# reused across the index and inference phases of the same run.
_MODEL_CACHE: Dict[Any, Any] = {}


def _cached(key: Any, build: Any) -> Any:
    instance = _MODEL_CACHE.get(key)
    if instance is None:
        instance = build()
        _MODEL_CACHE[key] = instance
    return instance


class EmbedderFactory:
    @staticmethod
    def create_dense(settings: Settings) -> DenseEmbedder:
        return _cached(
            ("dense", settings.dense_model_name, settings.device),
            lambda: HalongDenseEmbedder(settings.dense_model_name, settings.device),
        )

    @staticmethod
    def create_sparse(settings: Settings) -> SparseEmbedder:
        return _cached(
            ("sparse", settings.sparse_model_name),
            lambda: Bm25SparseEmbedder(settings.sparse_model_name),
        )


class QdrantClientFactory:
    @staticmethod
    def create(settings: Settings) -> Any:
        from qdrant_client import QdrantClient

        if settings.qdrant_url:
            try:
                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    prefer_grpc=False,
                    timeout=30.0,
                )
                client.get_collections()
                print(f"🗂️ [qdrant] connected to server {settings.qdrant_url}", flush=True)
                return client
            except Exception as error:
                print(
                    f"🗂️ [qdrant] server {settings.qdrant_url} unreachable ({error}); "
                    f"falling back to embedded local mode at {settings.qdrant_local_path}",
                    flush=True,
                )
        else:
            print(
                f"🗂️ [qdrant] no server url configured; using embedded local mode at "
                f"{settings.qdrant_local_path}",
                flush=True,
            )
        settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_local_path))


class LawRepository(Protocol):
    def ensure_collection(self, recreate: bool) -> None: ...

    def index(self, articles: Sequence[LawArticle], batch_size: int) -> int: ...

    def hybrid_search(
        self, query: str, limit: int, law_ids: Optional[Sequence[str]]
    ) -> List[LawReference]: ...


class QdrantLawRepository:
    _DENSE_VECTOR = "dense"
    _SPARSE_VECTOR = "bm25"

    def __init__(
        self,
        client: Any,
        collection_name: str,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._dense = dense_embedder
        self._sparse = sparse_embedder

    @property
    def client(self) -> Any:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection

    def _supports_payload_index(self) -> bool:
        inner = getattr(self._client, "_client", None)
        return type(inner).__name__ != "QdrantLocal"

    def ensure_collection(self, recreate: bool) -> None:
        from qdrant_client import models

        exists = self._client.collection_exists(self._collection)
        if exists and not recreate:
            return
        if exists:
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                self._DENSE_VECTOR: models.VectorParams(
                    size=self._dense.dimension,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=32, ef_construct=256),
                )
            },
            sparse_vectors_config={
                self._SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        if not self._supports_payload_index():
            print(
                "🗂️ [qdrant] embedded local mode: skipping payload index creation "
                "(filtering still works; use server Qdrant for filterable HNSW)",
                flush=True,
            )
            return
        self._client.create_payload_index(
            self._collection, "law_id", models.PayloadSchemaType.KEYWORD
        )
        self._client.create_payload_index(
            self._collection, "aid", models.PayloadSchemaType.INTEGER
        )

    def count(self) -> int:
        return int(self._client.count(self._collection, exact=True).count)

    @staticmethod
    def _expand_chunks(articles: Sequence[LawArticle]) -> List[Dict[str, Any]]:
        """Split each article into <=LAW_CHUNK_MAX_CHARS chunks (recursive chunking).

        Long articles retrieve better as focused chunks; every chunk keeps its
        ``article_text`` (the full original) so retrieval can merge chunks back to a
        single article. Short articles yield exactly one chunk (unchanged behavior).
        """
        records: List[Dict[str, Any]] = []
        for article in articles:
            chunks = _recursive_chunk(article.text) or [article.text]
            for chunk_index, chunk_text in enumerate(chunks):
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"alqac-law:{article.law_id}:{article.aid}:{chunk_index}",
                    )
                )
                records.append(
                    {
                        "id": point_id,
                        "law_id": article.law_id,
                        "aid": article.aid,
                        "order": article.order,
                        "chunk_index": chunk_index,
                        "n_chunks": len(chunks),
                        "text": chunk_text,
                        "article_text": article.text,
                    }
                )
        return records

    def index(self, articles: Sequence[LawArticle], batch_size: int = 256) -> int:
        from qdrant_client import models

        records = self._expand_chunks(articles)
        print(
            f"📚 [index] {len(articles)} articles -> {len(records)} chunks "
            f"(max {LAW_CHUNK_MAX_CHARS} chars, overlap {LAW_CHUNK_OVERLAP})",
            flush=True,
        )
        total = 0
        progress = _progress(total=len(records), desc="index", unit="chunk")
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [record["text"] for record in batch]
            dense_vectors = self._dense.encode_documents(texts)
            sparse_vectors = self._sparse.encode_documents(texts)
            points: List[Any] = []
            for record, dense_vector, sparse_vector in zip(batch, dense_vectors, sparse_vectors):
                indices, values = sparse_vector
                points.append(
                    models.PointStruct(
                        id=record["id"],
                        vector={
                            self._DENSE_VECTOR: dense_vector,
                            self._SPARSE_VECTOR: models.SparseVector(
                                indices=indices, values=values
                            ),
                        },
                        payload={
                            "law_id": record["law_id"],
                            "aid": record["aid"],
                            "order": record["order"],
                            "chunk_index": record["chunk_index"],
                            "n_chunks": record["n_chunks"],
                            "text": record["text"],
                            "article_text": record["article_text"],
                        },
                    )
                )
            self._client.upsert(self._collection, points=points, wait=True)
            total += len(points)
            progress.update(len(points))
        progress.close()
        return total

    def hybrid_search(
        self,
        query: str,
        limit: int,
        law_ids: Optional[Sequence[str]] = None,
    ) -> List[LawReference]:
        from qdrant_client import models

        dense_vector = self._dense.encode_query(query)
        sparse_indices, sparse_values = self._sparse.encode_query(query)
        query_filter = self._build_filter(law_ids)
        # Points are chunks; several chunks may belong to one article, so over-fetch
        # then merge down to `limit` distinct articles.
        point_limit = max(limit * 4, 40)
        prefetch_limit = max(point_limit * 2, 40)
        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=self._DENSE_VECTOR,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using=self._SPARSE_VECTOR,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=point_limit,
            with_payload=True,
        )
        # Merge chunks back to articles: keep each article once at its best-ranked
        # chunk score, and expose the full original article text (article_text).
        merged: Dict[Tuple[str, int], LawReference] = {}
        for point in response.points:
            payload = point.payload or {}
            law_id = str(payload.get("law_id", ""))
            aid = int(payload.get("aid", -1))
            key = (law_id, aid)
            if key in merged:
                continue
            merged[key] = LawReference(
                aid=aid,
                law_id=law_id,
                text=str(payload.get("article_text") or payload.get("text", "")),
                score=float(point.score),
            )
            if len(merged) >= limit:
                break
        references = list(merged.values())
        if TRACE.enabled:
            TRACE.law_calls.append(
                {
                    "query": query,
                    "law_ids_filter": list(law_ids) if law_ids else None,
                    "n_results": len(references),
                    "results": [
                        {
                            "law_id": ref.law_id,
                            "aid": ref.aid,
                            "score": round(ref.score, 4),
                            "text": ref.text[:400],
                        }
                        for ref in references
                    ],
                }
            )
        return references

    @staticmethod
    def _build_filter(law_ids: Optional[Sequence[str]]) -> Optional[Any]:
        from qdrant_client import models

        if not law_ids:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="law_id", match=models.MatchAny(any=list(law_ids)))]
        )


class CorpusIndexer:
    def __init__(self, loader: CorpusLoader, repository: QdrantLawRepository) -> None:
        self._loader = loader
        self._repository = repository

    def run(self, recreate: bool) -> int:
        articles = self._loader.load()
        print(f"📚 [index] loaded {len(articles)} articles", flush=True)
        self._repository.ensure_collection(recreate=recreate)
        indexed = self._repository.index(articles)
        print(f"📚 [index] collection now holds {self._repository.count()} points", flush=True)
        return indexed


class LanguageModel(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class Throttle(Protocol):
    def wait(self) -> None: ...


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._last_call = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


class GlobalRateLimiter:
    _last_call: float = 0.0

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - GlobalRateLimiter._last_call
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        GlobalRateLimiter._last_call = time.monotonic()


class AzureLanguageModel:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_retries: int,
        rate_limiter: Throttle,
        retry_delay: float = 5.0,
    ) -> None:
        self._model_name = model_name
        self._max_retries = max_retries
        self._rate_limiter = rate_limiter
        self._retry_delay = retry_delay
        self._client = self._build_client(model_name, temperature)

    @staticmethod
    def _build_client(model_name: str, temperature: float) -> Any:
        from langchain_litellm import ChatLiteLLMRouter
        from litellm import Router

        model_list = [
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": model_name,
                    "base_model": "gpt-4o",
                    "api_key": os.environ.get("AZURE_API_KEY"),
                    "api_version": os.environ.get("AZURE_API_VERSION"),
                    "api_base": os.environ.get("AZURE_API_BASE"),
                    "timeout": 60 * 60,
                },
            }
        ]
        router = Router(model_list=model_list)
        return ChatLiteLLMRouter(router=router, model_name=model_name, temperature=temperature)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=user_prompt.strip()),
        ]
        last_error: Optional[Exception] = None
        attempts = max(1, self._max_retries)
        for attempt in range(attempts):
            self._rate_limiter.wait()
            try:
                response = self._client.invoke(messages)
                return self._to_text(getattr(response, "content", response))
            except Exception as error:
                last_error = error
                is_last = attempt + 1 >= attempts
                delay = self._retry_delay * (attempt + 1)
                print(
                    f"🤖 [llm] '{self._model_name}' error {attempt + 1}/{attempts}: {error}"
                    + ("" if is_last else f"; retry in {delay:.0f}s"),
                    flush=True,
                )
                if not is_last:
                    time.sleep(delay)
        raise RuntimeError(f"LLM failed after {attempts} attempts: {last_error}")

    @staticmethod
    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments = [
                fragment.get("text", "") if isinstance(fragment, dict) else str(fragment)
                for fragment in content
            ]
            return "".join(fragments)
        return str(content)


class OpenAICompatLanguageModel:
    """Talk to a self-hosted vLLM (or any OpenAI-compatible) chat endpoint.

    vLLM serves the OpenAI ``/v1/chat/completions`` schema, so a plain ``requests`` POST
    is enough — no extra SDK. Used to run a local model (e.g. qwen3-4b-legal-pretrain served
    as ``vietnamese-law``) instead of Azure, avoiding the per-call rate cost. Same
    ``complete()`` interface as ``AzureLanguageModel`` so the rest of the pipeline is agnostic.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str,
        temperature: float,
        max_retries: int,
        rate_limiter: Throttle,
        retry_delay: float = 5.0,
        timeout: float = 600.0,
    ) -> None:
        self._url = api_base.rstrip("/") + "/chat/completions"
        self._model_name = model_name
        self._api_key = api_key
        self._temperature = temperature
        self._max_retries = max_retries
        self._rate_limiter = rate_limiter
        self._retry_delay = retry_delay
        self._timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": self._temperature,
        }
        last_error: Optional[Exception] = None
        attempts = max(1, self._max_retries)
        for attempt in range(attempts):
            self._rate_limiter.wait()
            try:
                response = requests.post(
                    self._url, headers=headers, json=payload, timeout=self._timeout
                )
                response.raise_for_status()
                body = response.json()
                return str(body["choices"][0]["message"]["content"])
            except Exception as error:  # noqa: BLE001 - surface any transport/parse failure
                last_error = error
                is_last = attempt + 1 >= attempts
                delay = self._retry_delay * (attempt + 1)
                print(
                    f"🤖 [llm] vllm '{self._model_name}' error {attempt + 1}/{attempts}: {error}"
                    + ("" if is_last else f"; retry in {delay:.0f}s"),
                    flush=True,
                )
                if not is_last:
                    time.sleep(delay)
        raise RuntimeError(f"vLLM endpoint failed after {attempts} attempts: {last_error}")


class LanguageModelFactory:
    @staticmethod
    def create(settings: Settings) -> LanguageModel:
        if settings.llm_provider == LLM_PROVIDER_VLLM:
            LanguageModelFactory.ensure_vllm_reachable(settings)
        LanguageModelFactory.announce(settings)
        return LanguageModelFactory.make(settings, settings.llm_temperature)

    @staticmethod
    def announce(settings: Settings) -> None:
        if settings.llm_provider == LLM_PROVIDER_VLLM:
            target = f"{settings.vllm_model_name} @ {settings.vllm_api_base}"
        else:
            target = settings.llm_model_name
        print(
            f"🤖 [llm] provider={settings.llm_provider} model={target} "
            f"temperature={settings.llm_temperature} min_interval={settings.llm_min_interval}s",
            flush=True,
        )

    @staticmethod
    def ensure_vllm_reachable(settings: Settings) -> None:
        url = settings.vllm_api_base.rstrip("/") + "/models"
        headers = {}
        if settings.vllm_api_key and settings.vllm_api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {settings.vllm_api_key}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            served = [str(item.get("id", "")) for item in response.json().get("data", [])]
        except Exception as error:  # noqa: BLE001 - any transport failure means unusable endpoint
            raise RuntimeError(
                f"Cannot reach vLLM at {settings.vllm_api_base} ({error}).\n"
                f"Start the server first: bash src/scripts/qwen_3_4b_legal.sh"
            ) from error
        if settings.vllm_model_name not in served:
            raise RuntimeError(
                f"vLLM is running but does not serve model {settings.vllm_model_name!r}.\n"
                f"Available: {', '.join(served) or '(none)'}\n"
                f"Fix with --vllm_model_name <name>, or change --served-model-name when serving."
            )

    @staticmethod
    def make(settings: Settings, temperature: float) -> LanguageModel:
        """Build a model at a given temperature for the configured provider.

        Shared by the main pipeline model and the outcome-predictor model (which needs a
        non-zero temperature for self-consistency voting). The process-global rate limiter
        keeps a single cadence across every model instance.
        """
        limiter = GlobalRateLimiter(settings.llm_min_interval)
        if settings.llm_provider == "vllm":
            return OpenAICompatLanguageModel(
                api_base=settings.vllm_api_base,
                model_name=settings.vllm_model_name,
                api_key=settings.vllm_api_key,
                temperature=temperature,
                max_retries=settings.llm_max_retries,
                rate_limiter=limiter,
                retry_delay=settings.llm_retry_delay,
            )
        return AzureLanguageModel(
            model_name=settings.llm_model_name,
            temperature=temperature,
            max_retries=settings.llm_max_retries,
            rate_limiter=limiter,
            retry_delay=settings.llm_retry_delay,
        )


class JsonResponseParser:
    @staticmethod
    def parse_object(raw: str) -> Dict[str, Any]:
        candidate = JsonResponseParser._extract(raw)
        if candidate is None:
            return {}
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract(raw: str) -> Optional[str]:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        text = fenced.group(1) if fenced else raw
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        return text[start : end + 1]


def _string_list(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    items: List[str] = []
    for element in value:
        text = str(element).strip()
        if text:
            items.append(text)
    return items[:limit]


_NUMERIC_TOKEN = re.compile(r"\d+(?:[.,/\-]\d+)*")


def _grounded(query: str, source: str) -> bool:
    """True unless the query invents a numeric specific (contract no., date, amount,
    article no.) absent from the source case text.

    At query-generation time the only legitimate source of concrete numbers is the
    case description itself — the model has not seen any judgment segment yet. So any
    number/date/id in a generated query that does not appear verbatim in ``source`` is
    a hallucination (e.g. an invented "hợp đồng số 01/2023") and the query is dropped.
    """
    for token in _NUMERIC_TOKEN.findall(query):
        if token not in source:
            return False
    return True


def _filter_grounded(queries: Sequence[str], source: str) -> List[str]:
    kept: List[str] = []
    for query in queries:
        if _grounded(query, source):
            kept.append(query)
        else:
            print(f"🚫 [query-guard] dropped hallucinated query: {query!r}", flush=True)
    return kept


_QUERY_STOPWORDS = {
    "của", "về", "các", "và", "cho", "một", "khi", "đã", "là", "trong", "có", "với",
    "theo", "được", "để", "hay", "hoặc", "những", "này", "đó", "số", "từ",
}


def _content_words(query: str) -> frozenset:
    words = re.findall(r"[0-9a-zA-ZÀ-ỹ]+", query.lower())
    return frozenset(w for w in words if w not in _QUERY_STOPWORDS)


def _dedupe_similar(queries: Sequence[str], threshold: float = 0.65) -> List[str]:
    """Drop near-duplicate queries by content-word Jaccard overlap.

    Two probes like "Nhận định của HĐXX về nội dung tranh chấp" and "HĐXX lập luận
    nội dung tranh chấp" hit the same segment, wasting an API call; keeping only the
    first cuts dry calls without losing coverage.
    """
    kept: List[str] = []
    kept_sets: List[frozenset] = []
    for query in queries:
        words = _content_words(query)
        if not words:
            continue
        duplicate = False
        for prior in kept_sets:
            union = words | prior
            if union and len(words & prior) / len(union) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(query)
            kept_sets.append(words)
    return kept


LAW_CHUNK_MAX_CHARS: int = 800
LAW_CHUNK_OVERLAP: int = 120
_CHUNK_SEPARATORS: List[str] = ["\n\n", "\n", ". ", "; ", ": ", ", ", " ", ""]

_SPLITTER_CACHE: Dict[Tuple[int, int], Any] = {}


def _get_splitter(max_chars: int, overlap: int) -> Any:
    key = (max_chars, overlap)
    splitter = _SPLITTER_CACHE.get(key)
    if splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            length_function=len,
            separators=_CHUNK_SEPARATORS,
            keep_separator=True,
            is_separator_regex=False,
        )
        _SPLITTER_CACHE[key] = splitter
    return splitter


def _recursive_chunk(
    text: str,
    max_chars: int = LAW_CHUNK_MAX_CHARS,
    overlap: int = LAW_CHUNK_OVERLAP,
) -> List[str]:
    """Recursive character chunking via LangChain's RecursiveCharacterTextSplitter.

    Long statute articles retrieve better as focused, overlapping chunks. Short
    articles fit under ``max_chars`` and return as a single chunk. The splitter walks
    the separator list from coarsest to finest, only descending when a piece is still
    too long, and repeats ``overlap`` characters between adjacent chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    return [chunk for chunk in _get_splitter(max_chars, overlap).split_text(text) if chunk.strip()]


class CaseQueryDecomposer:
    _SYSTEM = (
        "Bạn là chuyên gia phân tích tranh chấp pháp lý Việt Nam. "
        "Nhiệm vụ: phân rã yêu cầu vụ án thành từ khóa pháp lý và các truy vấn con để tìm kiếm."
    )

    def __init__(self, model: LanguageModel, parser: JsonResponseParser) -> None:
        self._model = model
        self._parser = parser

    def decompose(self, case: LegalCase) -> CaseQueryDecomposition:
        user_prompt = (
            f"{case.context()}\n\n"
            "Hãy trích xuất:\n"
            "1. case_query_keywords: các từ/cụm từ khóa pháp lý cốt lõi (quan hệ pháp luật, "
            "loại tranh chấp, yêu cầu của các bên).\n"
            "2. sub_queries: 3-6 truy vấn con ngắn gọn dùng để tìm điều luật liên quan.\n\n"
            "CHỈ dùng thông tin có trong mô tả vụ án. TUYỆT ĐỐI KHÔNG bịa số hợp đồng, ngày tháng, "
            "số tiền hay số điều luật không xuất hiện trong mô tả.\n\n"
            "Chỉ trả về JSON: "
            '{"case_query_keywords": ["..."], "sub_queries": ["..."]}'
        )
        payload = self._parser.parse_object(self._model.complete(self._SYSTEM, user_prompt))
        keywords = _string_list(payload.get("case_query_keywords"), 12)
        sub_queries = _filter_grounded(_string_list(payload.get("sub_queries"), 6), case.case_query)
        if not sub_queries:
            sub_queries = [case.case_query] if case.case_query else keywords[:3]
        return CaseQueryDecomposition(keywords=keywords, sub_queries=sub_queries)


class RetrievalAssessor:
    _SYSTEM = (
        "Bạn là chuyên gia thẩm định độ đầy đủ của căn cứ pháp lý cho một vụ án. "
        "Đánh giá xem các điều luật đã truy vấn có đủ để phán quyết hay chưa."
    )

    def __init__(self, model: LanguageModel, parser: JsonResponseParser) -> None:
        self._model = model
        self._parser = parser

    def assess(self, case: LegalCase, references: Sequence[LawReference]) -> RetrievalAssessment:
        summary = _format_law_references(references, max_chars=600)
        user_prompt = (
            f"{case.context()}\n\n"
            f"Các điều luật đã truy vấn:\n{summary}\n\n"
            "Đánh giá độ đầy đủ của căn cứ pháp lý để dự đoán kết quả vụ án. "
            "Nếu còn thiếu, chỉ ra thông tin còn thiếu và các từ khóa pháp lý mới "
            "(trích từ khái niệm mơ hồ hoặc điều luật đã tìm được) để tiếp tục tìm kiếm.\n\n"
            "Chỉ trả về JSON: "
            '{"sufficient": true/false, "missing_information": ["..."], '
            '"additional_keywords": ["..."]}'
        )
        payload = self._parser.parse_object(self._model.complete(self._SYSTEM, user_prompt))
        return RetrievalAssessment(
            sufficient=bool(payload.get("sufficient", False)),
            missing_information=_string_list(payload.get("missing_information"), 8),
            additional_keywords=_string_list(payload.get("additional_keywords"), 8),
        )


class LawRetrievalAgent:
    def __init__(
        self,
        repository: QdrantLawRepository,
        assessor: RetrievalAssessor,
        search_limit: int,
        max_iterations: int,
        max_evidence: int,
    ) -> None:
        self._repository = repository
        self._assessor = assessor
        self._search_limit = search_limit
        self._max_iterations = max_iterations
        self._max_evidence = max_evidence

    def retrieve(
        self, case: LegalCase, decomposition: CaseQueryDecomposition
    ) -> List[LawReference]:
        collected: Dict[Tuple[str, int], LawReference] = {}
        queries = list(dict.fromkeys(decomposition.sub_queries + decomposition.keywords))
        for iteration in range(self._max_iterations):
            self._search_and_merge(queries, collected)
            ranked = self._rank(collected)
            print(
                f"📖 [law] case={case.case_id} iter={iteration + 1} "
                f"queries={len(queries)} collected={len(collected)}",
                flush=True,
            )
            if iteration + 1 >= self._max_iterations:
                break
            assessment = self._assessor.assess(case, ranked[: self._max_evidence])
            if assessment.sufficient:
                break
            queries = self._next_queries(assessment)
            if not queries:
                break
        return self._rank(collected)[: self._max_evidence]

    def _search_and_merge(
        self, queries: Sequence[str], collected: Dict[Tuple[str, int], LawReference]
    ) -> None:
        for query in queries:
            for reference in self._repository.hybrid_search(query, self._search_limit):
                key = (reference.law_id, reference.aid)
                current = collected.get(key)
                if current is None or reference.score > current.score:
                    collected[key] = reference

    def _next_queries(self, assessment: RetrievalAssessment) -> List[str]:
        return list(
            dict.fromkeys(assessment.additional_keywords + assessment.missing_information)
        )

    @staticmethod
    def _rank(collected: Dict[Tuple[str, int], LawReference]) -> List[LawReference]:
        return sorted(collected.values(), key=lambda reference: reference.score, reverse=True)


class CaseContentClient:
    def __init__(
        self,
        api_url: str,
        api_token: str,
        rate_limiter: Throttle,
        max_retries: int,
        rate_limit_delay: float,
        max_rate_limit_retries: int,
    ) -> None:
        self._api_url = api_url
        self._api_token = api_token
        self._rate_limiter = rate_limiter
        self._max_retries = max_retries
        self._rate_limit_delay = rate_limit_delay
        self._max_rate_limit_retries = max_rate_limit_retries

    def retrieve(self, query: str, case_id: str, timeout: float = 30.0) -> List[RetrievedChunk]:
        headers = {"X-API-Key": self._api_token, "Content-Type": "application/json"}
        payload = {"query": query, "case_id": case_id}
        error_attempts = 0
        rate_limit_hits = 0
        while error_attempts <= self._max_retries:
            self._rate_limiter.wait()
            try:
                response = requests.post(
                    self._api_url, headers=headers, json=payload, timeout=timeout
                )
            except requests.RequestException as error:
                error_attempts += 1
                if error_attempts > self._max_retries:
                    print(f"📡 [case-api] error for case={case_id}: {error}", flush=True)
                    return []
                continue
            if response.status_code == 429:
                rate_limit_hits += 1
                if rate_limit_hits > self._max_rate_limit_retries:
                    print(
                        f"📡 [case-api] giving up after {rate_limit_hits - 1} rate-limit hits "
                        f"for case={case_id}",
                        flush=True,
                    )
                    return []
                print(
                    f"📡 [case-api] 429 for case={case_id}; wait {self._rate_limit_delay:.0f}s "
                    f"({rate_limit_hits}/{self._max_rate_limit_retries})",
                    flush=True,
                )
                self._sleep(self._rate_limit_delay)
                continue
            if response.status_code != 200:
                transient = response.status_code in (408, 425, 500, 502, 503, 504)
                if transient:
                    rate_limit_hits += 1
                    if rate_limit_hits > self._max_rate_limit_retries:
                        print(
                            f"📡 [case-api] giving up after {rate_limit_hits - 1} transient "
                            f"errors (last http {response.status_code}) for case={case_id}",
                            flush=True,
                        )
                        return []
                    print(
                        f"📡 [case-api] http {response.status_code} (transient) for case={case_id}; "
                        f"retry in {self._rate_limit_delay:.0f}s "
                        f"({rate_limit_hits}/{self._max_rate_limit_retries})",
                        flush=True,
                    )
                    self._sleep(self._rate_limit_delay)
                    continue
                print(f"📡 [case-api] http {response.status_code} for case={case_id}", flush=True)
                return []
            try:
                body = response.json()
            except ValueError:
                print(f"📡 [case-api] non-json 200 for case={case_id}", flush=True)
                return []
            chunks = self._parse(body)
            if TRACE.enabled:
                TRACE.case_calls.append(
                    {
                        "query": query,
                        "n_results": len(chunks),
                        "results": [
                            {
                                "chunk_id": chunk.chunk_id,
                                "score": round(chunk.score, 4),
                                "text": chunk.text[:400],
                            }
                            for chunk in chunks
                        ],
                    }
                )
            return chunks
        return []

    @staticmethod
    def _sleep(seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    @staticmethod
    def _parse(body: Any) -> List[RetrievedChunk]:
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            return []
        chunks: List[RetrievedChunk] = []
        for hit in results:
            if not isinstance(hit, dict):
                continue
            chunk_id = hit.get("chunk_id")
            if not isinstance(chunk_id, str):
                continue
            try:
                score = float(hit.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=str(hit.get("text", "")),
                    score=score,
                )
            )
        return chunks


class CaseKeywordExtractor:
    _SYSTEM = (
        "Bạn là chuyên gia trích xuất từ khóa để truy vấn tình tiết vụ án. "
        "Từ nội dung vụ án và điều luật liên quan, tạo các truy vấn ngắn giúp tìm đoạn văn bản "
        "chứng cứ trong bản án."
    )

    def __init__(self, model: LanguageModel, parser: JsonResponseParser) -> None:
        self._model = model
        self._parser = parser

    def extract(
        self, case: LegalCase, references: Sequence[LawReference], limit: int
    ) -> List[str]:
        law_summary = _format_law_references(references, max_chars=300)
        concat_corpus_query = f"{case.case_query}\n\nCăn cứ pháp lý liên quan:\n{law_summary}"
        user_prompt = (
            f"{concat_corpus_query}\n\n"
            f"Hãy tạo {limit} truy vấn con NGẮN (mỗi câu 4-10 từ, tiếng Việt) để truy xuất các đoạn "
            "bằng chứng trong bản án. Mỗi truy vấn nhắm ĐÚNG MỘT phần của bản án, và phải phủ đủ các "
            "phần sau (mỗi phần ít nhất một truy vấn):\n"
            "1) Phần QUYẾT ĐỊNH của Tòa (chấp nhận/bác yêu cầu, tuyên xử, buộc bồi thường).\n"
            "2) NHẬN ĐỊNH / lập luận của Hội đồng xét xử về nội dung tranh chấp.\n"
            "3) YÊU CẦU KHỞI KIỆN cụ thể của nguyên đơn (đối tượng, số tiền, diện tích).\n"
            "4) Ý KIẾN / trình bày phản bác của bị đơn.\n"
            "5) CHỨNG CỨ, tài liệu, hợp đồng, giấy tờ trong hồ sơ.\n"
            "6) Đoạn 'Căn cứ vào Điều... Bộ luật...' nêu các điều luật được áp dụng.\n"
            "CHỈ dùng tên người, địa danh, số hợp đồng, ngày tháng, con số ĐÚNG NHƯ xuất hiện trong "
            "mô tả vụ án ở trên. TUYỆT ĐỐI KHÔNG bịa ra số hợp đồng, ngày tháng, số tiền hay số điều "
            "luật không có trong mô tả (ví dụ không được tự chế 'hợp đồng số 01/2023').\n\n"
            'Chỉ trả về JSON: {"queries": ["..."]}'
        )
        payload = self._parser.parse_object(self._model.complete(self._SYSTEM, user_prompt))
        queries = _string_list(payload.get("queries"), limit)
        return _filter_grounded(queries, case.case_query)


class CaseEvidenceCollector:
    _DIVERSE_PROBE_IDX = (0, 9, 8, 1, 10, 11, 12, 6, 7, 13)
    _LAW_REF_RE = re.compile(r"(Điều|điều)\s+\d+|Nghị\s*định\s+\d+|Bộ\s*luật|\bLuật\s+[A-ZĐ]")

    @classmethod
    def _is_law_ref_query(cls, query: str) -> bool:
        return bool(cls._LAW_REF_RE.search(query))

    def __init__(
        self,
        client: CaseContentClient,
        extractor: CaseKeywordExtractor,
        probe_queries: Sequence[str],
        max_calls: int,
    ) -> None:
        self._client = client
        self._extractor = extractor
        self._probe_queries = list(probe_queries)
        self._max_calls = max_calls
        self.last_stats: Dict[str, int] = {"api_calls": 0, "dry_calls": 0, "segments": 0}

    def collect(
        self,
        case: LegalCase,
        decomposition: CaseQueryDecomposition,
        references: Sequence[LawReference],
    ) -> List[RetrievedChunk]:
        ordered = self._build_queries(case, decomposition, references)
        collected: Dict[str, RetrievedChunk] = {}
        calls = 0
        dry = 0
        dry_total = 0
        for query in ordered:
            if calls >= self._max_calls:
                break
            if calls >= 2 * max(len(collected), PENALTY_SAFE_MIN_SEGMENTS):
                break
            calls += 1
            before = len(collected)
            for chunk in self._client.retrieve(query, case.case_id):
                current = collected.get(chunk.chunk_id)
                if current is None or chunk.score > current.score:
                    collected[chunk.chunk_id] = chunk
            if len(collected) == before:
                dry += 1
                dry_total += 1
                if dry >= RETRIEVAL_DRY_STREAK:
                    break
            else:
                dry = 0
        self.last_stats = {
            "api_calls": calls,
            "dry_calls": dry_total,
            "segments": len(collected),
        }
        return sorted(collected.values(), key=lambda chunk: chunk.score, reverse=True)

    def _build_queries(
        self,
        case: LegalCase,
        decomposition: CaseQueryDecomposition,
        references: Sequence[LawReference],
    ) -> List[str]:
        spread = self._spread_queries(case.full_text)
        if spread:
            candidates = spread + list(decomposition.sub_queries) + list(self._probe_queries)
        else:
            keyword_queries = [q for q in self._extractor.extract(case, references, self._max_calls) if not self._is_law_ref_query(q)]
            probes = list(self._probe_queries)
            diverse_probes = [probes[i] for i in self._DIVERSE_PROBE_IDX if i < len(probes)]
            sub_queries = [q for q in decomposition.sub_queries if not self._is_law_ref_query(q)]
            interleaved = [
                q
                for pair in itertools.zip_longest(keyword_queries, diverse_probes)
                for q in pair
                if q
            ]
            candidates = interleaved + sub_queries + (
                [case.case_query] if case.case_query else []
            )
        ordered = list(dict.fromkeys(query.strip() for query in candidates if query.strip()))
        ordered = _dedupe_similar(ordered)
        if not ordered and case.case_query:
            ordered = [case.case_query]
        return ordered

    @staticmethod
    def _spread_queries(text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        step = max(len(words) // SPREAD_QUERY_COUNT, 1)
        queries: List[str] = []
        for start in range(0, len(words), step):
            window = " ".join(words[start : start + SPREAD_QUERY_WINDOW]).strip()
            if window:
                queries.append(window[:180])
            if len(queries) >= SPREAD_QUERY_COUNT:
                break
        return queries


class OutcomePredictor(Protocol):
    def predict(
        self,
        case: LegalCase,
        references: Sequence[LawReference],
        evidence: Sequence[RetrievedChunk],
    ) -> OutcomeReasoning: ...


class LlmOutcomePredictor:


    _SYSTEM = (
        "Bạn là thẩm phán mô phỏng của Tòa án Việt Nam. Các đoạn trích dưới đây được lấy từ CHÍNH "
        "bản án của vụ việc (phần Nhận định và Quyết định của Hội đồng xét xử). Hãy đọc kỹ, bám sát "
        "văn bản, KHÔNG suy đoán chủ quan, để xác định kết quả cho nguyên đơn (bên A) so với bị đơn (bên B)."
    )

    _MAX_EVIDENCE_CHUNKS = 14
    _CHUNK_CHARS = 700

    _RERANK_QUERY = (
        "Quyết định của Hội đồng xét xử: chấp nhận hay bác yêu cầu khởi kiện của nguyên đơn, "
        "toàn bộ hay một phần"
    )

    def __init__(
        self,
        model: LanguageModel,
        parser: JsonResponseParser,
        samples: int = 1,
        reranker: Optional[Reranker] = None,
        topk: int = 8,
        min_score: float = 0.0,
    ) -> None:
        self._model = model
        self._parser = parser
        self._samples = max(1, samples)
        self._reranker = reranker
        self._topk = topk
        self._min_score = min_score

    def predict(
        self,
        case: LegalCase,
        references: Sequence[LawReference],
        evidence: Sequence[RetrievedChunk],
    ) -> OutcomeReasoning:
        user_prompt = self._build_prompt(case, references, evidence)
        if TRACE.enabled:
            TRACE.classification_prompt = user_prompt
        votes: List[str] = []
        rationale = ""
        for _ in range(self._samples):
            payload = self._parser.parse_object(self._model.complete(self._SYSTEM, user_prompt))
            label = str(payload.get("prediction", "")).strip().upper()
            if label in VALID_LABELS:
                votes.append(label)
                rationale = str(payload.get("rationale", rationale)).strip()[:300]
        if not votes:
            label = DEFAULT_PREDICTION
            confidence = 0.0
        else:
            counts = Counter(votes)
            label, top = counts.most_common(1)[0]
            confidence = top / len(votes)
        print(
            f"🔮 [predict] case={case.case_id} label={label} conf={confidence:.2f} "
            f"(votes={len(votes)}/{self._samples}, evidence={len(evidence)})",
            flush=True,
        )
        return OutcomeReasoning(
            prediction=label,
            confidence=confidence,
            rationale=rationale or f"llm:{label}",
        )

    def _build_prompt(
        self,
        case: LegalCase,
        references: Sequence[LawReference],
        evidence: Sequence[RetrievedChunk],
    ) -> str:
        if self._reranker is not None:
            chunks = self._reranker.order(
                self._RERANK_QUERY, evidence, top_k=self._topk, min_score=self._min_score
            )
        else:
            chunks = list(evidence)[: self._topk]
        evidence_text = (
            "\n\n".join(
                f"[{index}] {chunk.text[: self._CHUNK_CHARS].strip()}"
                for index, chunk in enumerate(chunks, start=1)
            )
            or "(không truy xuất được đoạn bản án nào)"
        )
        law_text = _format_law_references(references[:8], max_chars=160)
        return (
            f"Vai trò bên A: {case.a_role}. Vai trò bên B: {case.b_role}.\n\n"
            f"Mô tả vụ án (yêu cầu khởi kiện):\n{case.case_query}\n\n"
            f"Các đoạn trích từ bản án (đã truy xuất qua API):\n{evidence_text}\n\n"
            f"Điều luật liên quan (tham khảo):\n{law_text}\n\n"
            "PHÂN TÍCH THEO 3 BƯỚC — chỉ căn cứ phần QUYẾT ĐỊNH / 'Tuyên xử' (đánh số [1],[2]...):\n\n"
            "BƯỚC 1: Với TỪNG yêu cầu khởi kiện của nguyên đơn (bên A), xác định Tòa xử ra sao:\n"
            "  - 'Chấp nhận yêu cầu', 'Buộc bị đơn phải...', 'Công nhận...' = yêu cầu ĐƯỢC chấp nhận.\n"
            "  - 'KHÔNG chấp nhận', 'Bác yêu cầu', 'Đình chỉ' = yêu cầu BỊ TỪ CHỐI. "
            "CHÚ Ý KỸ chữ 'KHÔNG'/'BÁC' đứng trước 'yêu cầu của nguyên đơn' — đây là nguyên đơn THUA.\n\n"
            "BƯỚC 2: Tổng hợp:\n"
            "  - TẤT CẢ yêu cầu của nguyên đơn bị bác / không chấp nhận → nguyên đơn thua hoàn toàn.\n"
            "  - TẤT CẢ yêu cầu được chấp nhận → nguyên đơn thắng hoàn toàn.\n"
            "  - Có cái được, có cái bị bác → thắng một phần.\n\n"
            "BƯỚC 3: Chọn nhãn (KHÔNG mặc định thiên về bên nào):\n"
            "  - B_WIN: bị đơn thắng — Tòa BÁC / KHÔNG chấp nhận TOÀN BỘ yêu cầu của nguyên đơn.\n"
            "  - A_WIN: nguyên đơn thắng — Tòa chấp nhận TOÀN BỘ yêu cầu, không bác phần nào.\n"
            "  - PARTIAL_A_WIN: thắng một phần, nguyên đơn được chấp nhận TRÊN 50% yêu cầu.\n"
            "  - PARTIAL_B_WIN: thắng một phần, nguyên đơn được chấp nhận KHÔNG QUÁ 50% yêu cầu.\n\n"
            "Nếu vụ án đình chỉ / công nhận hòa giải thành thì bám theo nội dung các bên thỏa thuận cuối cùng.\n"
            "Chọn ĐÚNG MỘT nhãn: A_WIN, PARTIAL_A_WIN, PARTIAL_B_WIN, B_WIN.\n"
            'Chỉ trả về JSON: {"prediction": "NHÃN", "rationale": "trích nguyên văn câu quyết định"}'
        )


class ClassifierOutcomePredictor:
    def __init__(self, model_id: str, device: str, max_length: int) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = device
        self._max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.to(device)
        self._model.eval()
        self._id2label = {
            int(key): str(value) for key, value in self._model.config.id2label.items()
        }

    def predict(
        self,
        case: LegalCase,
        references: Sequence[LawReference],
        evidence: Sequence[RetrievedChunk],
    ) -> OutcomeReasoning:
        text = " ".join((case.case_query or case.context()).split()).strip()
        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            probs = self._model(**inputs).logits.softmax(dim=-1)[0]
        predicted_id = int(probs.argmax())
        label = self._id2label.get(predicted_id, "B_WIN")
        if label not in VALID_LABELS:
            label = "B_WIN"
        confidence = float(probs[predicted_id])
        print(
            f"🔮 [predict] case={case.case_id} label={label} conf={confidence:.2f}",
            flush=True,
        )
        return OutcomeReasoning(prediction=label, confidence=confidence, rationale=f"classifier:{label}")


def _format_law_references(references: Sequence[LawReference], max_chars: int) -> str:
    if not references:
        return "(chưa có)"
    lines: List[str] = []
    for index, reference in enumerate(references, start=1):
        snippet = reference.text[:max_chars].replace("\n", " ").strip()
        lines.append(
            f"[{index}] {reference.law_id} | Điều aid={reference.aid} "
            f"(score={reference.score:.3f}): {snippet}"
        )
    return "\n".join(lines)


def _format_chunks(chunks: Sequence[RetrievedChunk], max_chars: int) -> str:
    if not chunks:
        return "(chưa có)"
    lines: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        snippet = chunk.text[:max_chars].replace("\n", " ").strip()
        lines.append(f"[{index}] {chunk.chunk_id} (score={chunk.score:.3f}): {snippet}")
    return "\n".join(lines)


class CorpusLawIndex:

    def __init__(self, corpus_path: Path) -> None:
        self._number_to_aid: Dict[Tuple[str, int], int] = {}
        self._law_ids: set = set()
        with corpus_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        for law in records:
            law_id = str(law.get("law_id", "")).strip()
            self._law_ids.add(law_id)
            article_no = 0
            for entry in law.get("content", []):
                try:
                    aid = int(entry.get("aid"))
                except (TypeError, ValueError):
                    continue
                article_no += 1
                self._number_to_aid[(law_id, article_no)] = aid

    @property
    def law_ids(self) -> set:
        return self._law_ids

    def resolve_number(self, law_id: str, article_no: int) -> Optional[int]:
        return self._number_to_aid.get((law_id, article_no))


class LawNameResolver:

    _CODE_PATTERN = re.compile(r"\d+/\d{4}/[A-Za-zĐđ\-]+")
    _OUTDATED_YEARS: Tuple[str, ...] = (
        "1987", "1993", "1995", "1998", "2000", "2003", "2004", "2005", "2006", "2009",
    )

    def __init__(self, corpus_law_ids: set) -> None:
        self._corpus_law_ids = corpus_law_ids

    def resolve(self, name: str) -> Optional[str]:
        code = self._match_code(name)
        if code is not None:
            return code
        return self._match_keywords(name.lower())

    def _match_code(self, name: str) -> Optional[str]:
        for match in self._CODE_PATTERN.finditer(name):
            candidate = match.group(0)
            if candidate in self._corpus_law_ids:
                return candidate
        return None

    def _match_keywords(self, lowered: str) -> Optional[str]:
        if "tố tụng dân sự" in lowered:
            return "92/2015/QH13"
        if "tố tụng hành chính" in lowered:
            return "93/2015/QH13"
        if "dân sự" in lowered:
            return None if self._is_outdated(lowered) else "91/2015/QH13"
        if "hình sự" in lowered:
            return "100/2015/QH13"
        if "đất đai" in lowered:
            return None if self._is_outdated(lowered) else "45/2013/QH13"
        if "hôn nhân" in lowered:
            return None if self._is_outdated(lowered) else "52/2014/QH13"
        if "án phí" in lowered or "lệ phí" in lowered:
            return None if "pháp lệnh" in lowered else "326/2016/UBTVQH14"
        if "thi hành án" in lowered:
            return "26/2008/QH12"
        if "hộ tịch" in lowered:
            return "60/2014/QH13"
        if "tổ chức tín dụng" in lowered:
            return "47/2010/QH12"
        if "kinh doanh bất động sản" in lowered:
            return None if self._is_outdated(lowered) else "66/2014/QH13"
        if "xây dựng" in lowered:
            return "50/2014/QH13"
        return None

    def _is_outdated(self, lowered: str) -> bool:
        return any(year in lowered for year in self._OUTDATED_YEARS)


class CitationLawExtractor:

    _ART = re.compile(r"[Đđ]iều\s+0*(\d{1,3})")
    _LAWNAME = re.compile(
        r"(Bộ luật Dân sự(?:\s+năm\s+\d{4})?"
        r"|Bộ luật Tố tụng dân sự(?:\s+năm\s+\d{4})?"
        r"|Bộ luật Hình sự(?:\s+năm\s+\d{4})?"
        r"|Luật Tố tụng hành chính(?:\s+năm\s+\d{4})?"
        r"|Luật Hôn nhân và gia đình(?:\s+năm\s+\d{4})?"
        r"|Luật Đất đai(?:\s+năm\s+\d{4})?"
        r"|Luật Hộ tịch(?:\s+năm\s+\d{4})?"
        r"|Luật Xây dựng(?:\s+năm\s+\d{4})?"
        r"|Nghị quyết số?[:\s]*\d+/\d{4}/[A-Za-zĐđ\-]+"
        r"|\d+/\d{4}/[A-Za-zĐđ\-]+)",
        re.IGNORECASE,
    )
    _ALLOWED_WORDS = {
        "căn", "cứ", "áp", "dụng", "các", "điều", "khoản", "điểm", "và", "của", "số",
    }
    _TAIL_MAX_WORDS = 40

    def __init__(self, index: CorpusLawIndex, resolver: LawNameResolver) -> None:
        self._index = index
        self._resolver = resolver

    def extract(self, evidence: Sequence[RetrievedChunk]) -> List[LawReference]:
        text = "\n".join(chunk.text for chunk in evidence if chunk.text)
        if not text:
            return []
        keys: List[Tuple[str, int]] = []
        seen: set = set()
        for match in self._LAWNAME.finditer(text):
            law_id = self._resolver.resolve(match.group(0))
            if not law_id:
                continue
            span = self._citation_tail(text[: match.start()])
            for art in self._ART.finditer(span):
                aid = self._index.resolve_number(law_id, int(art.group(1)))
                if aid is None:
                    continue
                key = (law_id, aid)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return [LawReference(aid=aid, law_id=law_id, score=1.0) for law_id, aid in keys]

    def _citation_tail(self, preceding: str) -> str:
        words = preceding.split()
        index = len(words)
        while index > 0 and (len(words) - index) < self._TAIL_MAX_WORDS and self._is_citation_word(words[index - 1]):
            index -= 1
        return " ".join(words[index:])

    def _is_citation_word(self, word: str) -> bool:
        cleaned = word.strip(".,;:()").lower()
        if cleaned == "" or cleaned.isdigit():
            return True
        if cleaned in self._ALLOWED_WORDS:
            return True
        if re.fullmatch(r"[a-đ]", cleaned):
            return True
        if re.fullmatch(r"\d+/\d{4}/[a-zđ\-]+", cleaned):
            return True
        return False


class LawEvidenceSelector:

    _SYSTEM = (
        "Bạn là chuyên gia pháp lý Việt Nam. Từ danh sách điều luật ứng viên đã truy xuất, hãy CHỌN LỌC "
        "chỉ những điều luật mà Tòa án THỰC SỰ áp dụng/viện dẫn để giải quyết vụ án, loại bỏ điều không liên quan."
    )

    def __init__(
        self,
        model: LanguageModel,
        parser: JsonResponseParser,
        target: int,
    ) -> None:
        self._model = model
        self._parser = parser
        self._target = max(1, target)

    def select(
        self,
        case: LegalCase,
        references: Sequence[LawReference],
        evidence: Sequence[RetrievedChunk],
    ) -> List[LawReference]:
        candidates = list(references)
        if len(candidates) <= 1:
            return candidates
        catalog = "\n".join(
            f"[{index}] {ref.law_id} | Điều aid={ref.aid}: "
            f"{ref.text[:220].replace(chr(10), ' ').strip()}"
            for index, ref in enumerate(candidates, start=1)
        )
        decision = "\n".join(f"- {chunk.text[:400]}" for chunk in list(evidence)[:6]) or "(chưa có)"
        user_prompt = (
            f"Mô tả vụ án:\n{case.case_query}\n\n"
            f"Trích đoạn bản án (để đối chiếu điều luật Tòa viện dẫn):\n{decision}\n\n"
            f"Danh sách điều luật ứng viên:\n{catalog}\n\n"
            f"Hãy chọn các điều luật (theo số thứ tự trong ngoặc vuông) mà Tòa án nhiều khả năng đã áp dụng, "
            f"ưu tiên độ CHÍNH XÁC hơn số lượng. Chọn tối đa {self._target} điều, loại bỏ điều lạc đề.\n"
            'Chỉ trả về JSON: {"selected": [<số thứ tự>, ...]}'
        )
        payload = self._parser.parse_object(self._model.complete(self._SYSTEM, user_prompt))
        raw = payload.get("selected")
        if not isinstance(raw, list):
            return candidates[: self._target]
        chosen: List[LawReference] = []
        seen: set = set()
        for item in raw:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(candidates) and idx not in seen:
                seen.add(idx)
                chosen.append(candidates[idx - 1])
        if not chosen:
            return candidates[: self._target]
        return chosen[: self._target]


class SubmissionBuilder:
    def __init__(
        self,
        max_law_evidence: int,
        law_selector: Optional[LawEvidenceSelector] = None,
        citation_extractor: Optional[CitationLawExtractor] = None,
        reranker: Optional[Reranker] = None,
        law_topk: int = 8,
        law_min_score: float = 0.0,
    ) -> None:
        self._max_law_evidence = max_law_evidence
        self._law_selector = law_selector
        self._citation_extractor = citation_extractor
        self._reranker = reranker
        self._law_topk = law_topk
        self._law_min_score = law_min_score

    def build(
        self,
        case: LegalCase,
        references: Sequence[LawReference],
        evidence: Sequence[RetrievedChunk],
        reasoning: OutcomeReasoning,
    ) -> SubmissionEntry:
        selected = self._select_law_evidence(case, references, evidence)
        law_evidence: List[Dict[str, Any]] = []
        seen: set = set()
        for reference in selected:
            key = (reference.law_id, reference.aid)
            if key in seen:
                continue
            seen.add(key)
            law_evidence.append(reference.as_evidence())
        return SubmissionEntry(
            case_id=case.case_id,
            case_evidence=[chunk.chunk_id for chunk in evidence],
            law_evidence=law_evidence,
            prediction=reasoning.prediction,
        )

    def _select_law_evidence(
        self,
        case: LegalCase,
        references: Sequence[LawReference],
        evidence: Sequence[RetrievedChunk],
    ) -> List[LawReference]:
 
        cited: List[LawReference] = []
        if self._citation_extractor is not None:
            cited = self._citation_extractor.extract(evidence)
        if self._law_selector is not None:
            selected = self._law_selector.select(
                case, references[: self._max_law_evidence], evidence
            )
        else:
            selected = list(references[: self._max_law_evidence])
        if self._reranker is not None and selected:
            selected = self._reranker.order(
                case.case_query, selected, top_k=self._law_topk, min_score=self._law_min_score
            )
        merged: List[LawReference] = []
        seen: set = set()
        for reference in list(cited) + list(selected):
            key = (reference.law_id, reference.aid)
            if key in seen:
                continue
            seen.add(key)
            merged.append(reference)
            if len(merged) >= self._max_law_evidence:
                break
        print(
            f"📖 [law] case={case.case_id} cited={len(cited)} selected={len(selected)} "
            f"-> submit={len(merged)}",
            flush=True,
        )
        return merged


class DeepLegalAgent:
    def __init__(
        self,
        decomposer: CaseQueryDecomposer,
        law_agent: LawRetrievalAgent,
        evidence_collector: CaseEvidenceCollector,
        predictor: OutcomePredictor,
        builder: SubmissionBuilder,
    ) -> None:
        self._decomposer = decomposer
        self._law_agent = law_agent
        self._evidence_collector = evidence_collector
        self._predictor = predictor
        self._builder = builder
        self.debug_rows: List[Dict[str, Any]] = []
        self.trace_entries: List[Dict[str, Any]] = []

    def run_case(self, case: LegalCase) -> SubmissionEntry:
        if TRACE.enabled:
            TRACE.reset()
        decomposition = self._decomposer.decompose(case)
        references = self._law_agent.retrieve(case, decomposition)
        evidence = self._evidence_collector.collect(case, decomposition, references)
        reasoning = self._predictor.predict(case, references, evidence)
        entry = self._builder.build(case, references, evidence, reasoning)
        stats = self._evidence_collector.last_stats
        row = {
            "case_id": case.case_id,
            "api_calls": stats.get("api_calls", 0),
            "dry_calls": stats.get("dry_calls", 0),
            "segments": len(entry.case_evidence),
            "law_evidence": len(entry.law_evidence),
            "prediction": entry.prediction,
        }
        self.debug_rows.append(row)
        print(
            f"⚖️ [case {case.case_id}] api={row['api_calls']} dry={row['dry_calls']} "
            f"seg={row['segments']} law={row['law_evidence']} pred={row['prediction']}",
            flush=True,
        )
        if TRACE.enabled:
            self.trace_entries.append(
                {
                    "case_id": case.case_id,
                    "case_query": case.case_query,
                    "a_role": case.a_role,
                    "b_role": case.b_role,
                    "prediction": entry.prediction,
                    "law_evidence_submitted": entry.law_evidence,
                    "case_evidence_submitted": entry.case_evidence,
                    "n_case_api_calls": len(TRACE.case_calls),
                    "n_unique_segments": len(entry.case_evidence),
                    "n_law_queries": len(TRACE.law_calls),
                    "evidence_fed_to_classifier": len(evidence),
                    "classification_prompt": TRACE.classification_prompt,
                    "unique_case_segments_retrieved": [
                        {"chunk_id": chunk.chunk_id, "score": round(chunk.score, 4), "text": chunk.text}
                        for chunk in evidence
                    ],
                    "case_api_calls": list(TRACE.case_calls),
                    "law_retrieval_calls": list(TRACE.law_calls),
                }
            )
        return entry

    def run_dataset(
        self,
        cases: Sequence[LegalCase],
        checkpoint: Optional[Callable[[List[SubmissionEntry]], None]] = None,
    ) -> List[SubmissionEntry]:
        entries: List[SubmissionEntry] = []
        progress = _progress(total=len(cases), desc="inference", unit="case")
        try:
            for case in cases:
                progress.set_description_str(f"[inference] {case.case_id:<14}", refresh=False)
                try:
                    entries.append(self.run_case(case))
                except Exception as error:
                    print(
                        f"⚠️ [case {case.case_id}] FAILED ({error}) -> fallback "
                        f"'{DEFAULT_PREDICTION}'",
                        flush=True,
                    )
                    entries.append(
                        SubmissionEntry(
                            case_id=case.case_id,
                            case_evidence=[],
                            law_evidence=[],
                            prediction=DEFAULT_PREDICTION,
                        )
                    )
                progress.update(1)
                if checkpoint is not None:
                    checkpoint(entries)
        finally:
            progress.close()
        return entries


class RepositoryFactory:
    @staticmethod
    def create(settings: Settings) -> QdrantLawRepository:
        client = QdrantClientFactory.create(settings)
        dense = EmbedderFactory.create_dense(settings)
        sparse = EmbedderFactory.create_sparse(settings)
        return QdrantLawRepository(client, settings.collection_name, dense, sparse)


class DeepLegalAgentFactory:
    @staticmethod
    def create(settings: Settings, repository: QdrantLawRepository) -> DeepLegalAgent:
        model = LanguageModelFactory.create(settings)
        parser = JsonResponseParser()
        reranker = (
            _cached(
                ("rerank", settings.rerank_model_name, settings.device),
                lambda: Reranker(settings.rerank_model_name, settings.device),
            )
            if settings.use_reranker
            else None
        )
        decomposer = CaseQueryDecomposer(model, parser)
        assessor = RetrievalAssessor(model, parser)
        law_agent = LawRetrievalAgent(
            repository=repository,
            assessor=assessor,
            search_limit=settings.law_search_limit,
            max_iterations=settings.max_retrieval_iterations,
            max_evidence=settings.max_law_evidence,
        )
        case_client = CaseContentClient(
            api_url=settings.case_api_url,
            api_token=settings.case_api_token,
            rate_limiter=RateLimiter(settings.api_min_interval),
            max_retries=settings.api_max_retries,
            rate_limit_delay=settings.api_rate_limit_delay,
            max_rate_limit_retries=settings.api_max_rate_limit_retries,
        )
        extractor = CaseKeywordExtractor(model, parser)
        evidence_collector = CaseEvidenceCollector(
            client=case_client,
            extractor=extractor,
            probe_queries=JUDGMENT_PROBE_QUERIES,
            max_calls=settings.max_case_queries,
        )
        predictor = DeepLegalAgentFactory._build_predictor(settings, model, parser, reranker)
        law_selector = (
            LawEvidenceSelector(model, parser, settings.law_evidence_target)
            if settings.law_select
            else None
        )
        corpus_index = CorpusLawIndex(settings.corpus_path)
        citation_extractor = CitationLawExtractor(
            corpus_index, LawNameResolver(corpus_index.law_ids)
        )
        builder = SubmissionBuilder(
            settings.max_law_evidence,
            law_selector=law_selector,
            citation_extractor=citation_extractor,
            reranker=reranker,
            law_topk=settings.law_rerank_topk,
            law_min_score=settings.law_rerank_min_score,
        )
        return DeepLegalAgent(decomposer, law_agent, evidence_collector, predictor, builder)

    @staticmethod
    def _build_predictor(
        settings: Settings,
        model: LanguageModel,
        parser: JsonResponseParser,
        reranker: Optional[Reranker] = None,
    ) -> OutcomePredictor:
        if settings.predictor_kind == "classifier":
            return ClassifierOutcomePredictor(
                model_id=settings.outcome_model_id,
                device=settings.device,
                max_length=settings.outcome_max_length,
            )

        if settings.outcome_samples > 1:
            predictor_model: LanguageModel = LanguageModelFactory.make(
                settings, settings.outcome_temperature
            )
        else:
            predictor_model = model
        return LlmOutcomePredictor(
            predictor_model,
            parser,
            samples=settings.outcome_samples,
            reranker=reranker,
        )


class CaseDatasetLoader:
    def __init__(self, input_path: Path) -> None:
        self._input_path = input_path

    def load(self, limit: Optional[int] = None) -> List[LegalCase]:
        with self._input_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        cases = [LegalCase.from_record(record) for record in records]
        if limit is not None:
            cases = cases[:limit]
        return cases


class SubmissionWriter:
    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def write(self, entries: Sequence[SubmissionEntry], quiet: bool = False) -> None:
        payload = [entry.as_dict() for entry in entries]
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._output_path.with_suffix(self._output_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp.replace(self._output_path)
        if not quiet:
            print(f"💾 [write] {len(payload)} entries -> {self._output_path}", flush=True)


class Application:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def index(self, recreate: bool) -> None:
        repository = RepositoryFactory.create(self._settings)
        loader = JsonLawCorpusLoader(self._settings.corpus_path)
        CorpusIndexer(loader, repository).run(recreate=recreate)

    def infer(self, limit: Optional[int]) -> None:
        repository = RepositoryFactory.create(self._settings)
        if not repository.client.collection_exists(self._settings.collection_name):
            raise RuntimeError(
                "Collection missing. Run indexing first: python deep_agents.py --index"
            )
        TRACE.enabled = bool(self._settings.trace_output)
        agent = DeepLegalAgentFactory.create(self._settings, repository)
        all_cases = CaseDatasetLoader(self._settings.input_path).load()
        processed = all_cases[:limit] if limit is not None else all_cases
        writer = SubmissionWriter(self._settings.output_path)

        def _covered(entries: List[SubmissionEntry]) -> List[SubmissionEntry]:
            done = {entry.case_id for entry in entries}
            filled = list(entries)
            for case in all_cases:
                if case.case_id not in done:
                    filled.append(
                        SubmissionEntry(
                            case_id=case.case_id,
                            case_evidence=[],
                            law_evidence=[],
                            prediction="",
                        )
                    )
            return filled

        def _checkpoint(entries: List[SubmissionEntry]) -> None:
            writer.write(_covered(entries), quiet=True)

        entries = agent.run_dataset(processed, checkpoint=_checkpoint)
        final = _covered(entries)
        writer.write(final)
        self._write_debug(agent.debug_rows)
        self._write_trace(agent.trace_entries)

    def _write_trace(self, entries: List[Dict[str, Any]]) -> None:
        if not self._settings.trace_output or not entries:
            return
        path = Path(self._settings.trace_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Single case -> one object (matches the reference output.json); many -> a list.
        payload: Any = entries[0] if len(entries) == 1 else entries
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📝 [trace] wrote {len(entries)} case trace(s) -> {path}", flush=True)

    def _write_debug(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        calls = [r["api_calls"] for r in rows]
        labels: Counter = Counter(r["prediction"] for r in rows)
        summary = {
            "n_cases": len(rows),
            "api_calls_mean": round(sum(calls) / len(calls), 2),
            "api_calls_max": max(calls),
            "dry_calls_total": sum(r["dry_calls"] for r in rows),
            "label_distribution": dict(labels),
            "rows": rows,
        }
        debug_path = self._settings.output_path.with_suffix(".debug.json")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"🐛 [debug] {len(rows)} cases | api_calls mean={summary['api_calls_mean']} "
            f"max={summary['api_calls_max']} dry_total={summary['dry_calls_total']} "
            f"-> {debug_path}",
            flush=True,
        )


# Config overrides exposed on the CLI, grouped by the module each one configures.
# Path-typed args are converted to Path in main(); the rest pass through as-is.
_PATH_OVERRIDES = ("input_path", "output_path", "corpus_path")
_VALUE_OVERRIDES = (
    # vector store: Qdrant
    "collection_name",
    "qdrant_url",
    "qdrant_local_path",
    # embedding models
    "dense_model_name",
    "sparse_model_name",
    # LLM
    "llm_provider",
    "llm_model_name",
    "llm_temperature",
    "llm_max_retries",
    "llm_min_interval",
    # self-hosted vLLM
    "vllm_api_base",
    "vllm_model_name",
    "vllm_api_key",
    # case content API
    "case_api_url",
    "api_min_interval",
    # law retrieval agent
    "law_search_limit",
    "max_retrieval_iterations",
    "max_law_evidence",
    # case evidence collector
    "max_case_queries",
    # outcome classifier
    "outcome_model_id",
    # outcome prediction strategy
    "predictor_kind",
    "outcome_samples",
    "outcome_temperature",
    # law evidence selection
    "law_select",
    "law_evidence_target",
    "use_reranker",
    "rerank_model_name",
    "law_rerank_topk",
    "law_rerank_min_score",
    "trace_output",
)
_OVERRIDE_TYPES: Dict[str, Any] = {
    "collection_name": str,
    "qdrant_url": str,
    "qdrant_local_path": str,
    "dense_model_name": str,
    "sparse_model_name": str,
    "llm_provider": str,
    "llm_model_name": str,
    "llm_temperature": float,
    "llm_max_retries": int,
    "llm_min_interval": float,
    "vllm_api_base": str,
    "vllm_model_name": str,
    "vllm_api_key": str,
    "case_api_url": str,
    "api_min_interval": float,
    "law_search_limit": int,
    "max_retrieval_iterations": int,
    "max_law_evidence": int,
    "max_case_queries": int,
    "outcome_model_id": str,
    "predictor_kind": str,
    "outcome_samples": int,
    "outcome_temperature": float,
    "law_select": str,
    "law_evidence_target": int,
    "use_reranker": str,
    "rerank_model_name": str,
    "law_rerank_topk": int,
    "law_rerank_min_score": float,
    "trace_output": str,
}


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALQAC deep legal agent")
    # Actions
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    # Runtime: GPU id(s) on the server, comma-separated (e.g. "0" or "0,1").
    # They are exposed via CUDA_VISIBLE_DEVICES; use "-1" (or "cpu") to force CPU.
    parser.add_argument("--gpu-ids", type=str, default=None)
    # Data / IO
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--corpus-path", type=str, default=None)
    # Config overrides (grouped by module)
    for name in _VALUE_OVERRIDES:
        parser.add_argument(f"--{name}", type=_OVERRIDE_TYPES[name], default=None)
    return parser.parse_args(list(argv))


def _resolve_gpu_ids(raw: Optional[str]) -> Optional[str]:
    """Map --gpu-ids into a torch device string and set CUDA_VISIBLE_DEVICES.

    Returns the device override (e.g. "cuda:0", "cpu") or None when unset.
    Restricting the visible GPUs re-indexes them from 0, so the primary
    compute device is always "cuda:0" once the mask is applied.
    """
    if raw is None:
        return None
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not ids or any(gid in ("-1", "cpu") for gid in ids):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if any(not gid.lstrip("+").isdigit() for gid in ids):
        raise ValueError(f"--gpu-ids expects integers or 'cpu', got: {raw!r}")
    # Match nvidia-smi indices on mixed-GPU hosts: without PCI_BUS_ID ordering CUDA
    # sorts by speed, so "5" could bind a different physical card than intended.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gid)) for gid in ids)
    return "cuda:0"


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    overrides: Dict[str, Any] = {}
    for name in _PATH_OVERRIDES:
        value = getattr(args, name)
        if value:
            overrides[name] = Path(value)
    for name in _VALUE_OVERRIDES:
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    device_override = _resolve_gpu_ids(args.gpu_ids)
    if device_override is not None:
        overrides["device"] = device_override
    settings = Settings.from_environment(overrides)
    application = Application(settings)
    # One process for both phases so embedders/reranker load once (see _MODEL_CACHE).
    if args.index or args.index_only:
        application.index(recreate=args.recreate)
        if args.index_only:
            return 0
    application.infer(limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
