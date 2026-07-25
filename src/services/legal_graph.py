from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

import jurist as da

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_GRAPH_DB_PATH = REPO_ROOT / ".cache" / "graph_db"
DEFAULT_COLLECTION_NAME = "legal_graph_store.pkl"

GRAPH_NODE_TYPE = "Cases"
RELATION_CITES = "CITES"

CITATION_SPAN = re.compile(
    r"((?:(?:kho[aả]n\s+\d+\s+)?[Đđ]i[eề]u\s+\d+[\s,;và]*)+)\s*(?:c[uủ]a\s+)?"
    r"((?:B[ộo]\s+lu[ậa]t|Lu[ậa]t|Ngh[ịi]\s+quy[ếe]t|Ngh[ịi]\s+đ[ịi]nh|Ph[áa]p\s+l[ệe]nh)[^;.\n]{0,50})"
)
ARTICLE_NUMBER = re.compile(r"[Đđ]i[eề]u\s+(\d+)")

SEED_SIZE: int = 10
EXPAND_TOP: int = 5
NEIGHBOR_HOPS: int = 2
KNN_TOP_K: int = 3


def _import_legalgraphrag() -> Tuple[Any, Any]:
    if "ALQAC" not in sys.modules:
        alias = types.ModuleType("ALQAC")
        alias.__path__ = [str(REPO_ROOT)]
        sys.modules["ALQAC"] = alias
    base = "ALQAC.src.core.LegalGraphRAG.core.graph_construct"
    graph_db = importlib.import_module(f"{base}.graph_db")
    feature_graph = importlib.import_module(f"{base}.feature_graph")
    return graph_db, feature_graph


_graph_db, _feature_graph = _import_legalgraphrag()
GraphDBManager = _graph_db.GraphDBManager
InMemoryGraphDB = _graph_db.InMemoryGraphDB


@dataclass(frozen=True)
class GraphLawReference:
    point_id: str
    law_id: str
    aid: int
    text: str
    score: float
    neighbors: Tuple[str, ...]

    def as_evidence(self) -> Dict[str, Any]:
        return {"law_id": self.law_id, "aid": self.aid}


class LawNameResolver:
    _CODE = re.compile(r"\d+/\d{4}/[A-Za-zĐđ\-]+")
    _OUTDATED = ("1987", "1993", "1995", "1998", "2000", "2003", "2004", "2005", "2006", "2009")

    def __init__(self, corpus_law_ids: Set[str]) -> None:
        self._ids = corpus_law_ids

    def resolve(self, name: str) -> Optional[str]:
        for match in self._CODE.finditer(name):
            if match.group(0) in self._ids:
                return match.group(0)
        low = name.lower()
        outdated = any(year in low for year in self._OUTDATED)
        if "tố tụng dân sự" in low:
            return "92/2015/QH13"
        if "tố tụng hành chính" in low:
            return "93/2015/QH13"
        if "dân sự" in low:
            return None if outdated else "91/2015/QH13"
        if "hình sự" in low:
            return "100/2015/QH13"
        if "đất đai" in low:
            return None if outdated else "45/2013/QH13"
        if "hôn nhân" in low:
            return None if outdated else "52/2014/QH13"
        if "án phí" in low or "lệ phí" in low:
            return None if "pháp lệnh" in low else "326/2016/UBTVQH14"
        if "thi hành án" in low:
            return "26/2008/QH12"
        if "hộ tịch" in low:
            return "60/2014/QH13"
        if "khiếu nại" in low:
            return None if ("tố cáo" in low or outdated) else "02/2011/QH13"
        if "tổ chức tín dụng" in low:
            return "47/2010/QH12"
        if "kinh doanh bất động sản" in low:
            return None if outdated else "66/2014/QH13"
        if "xây dựng" in low:
            return "50/2014/QH13"
        return None


class CitationGraphBuilder:
    def build(self, articles: Sequence[da.LawArticle]) -> Dict[str, List[str]]:
        """Build an undirected citation adjacency map by linking each article to the articles it references."""
        by_number: Dict[Tuple[str, int], da.LawArticle] = {}
        for article in articles:
            by_number[(article.law_id, article.order + 1)] = article
        resolver = LawNameResolver({article.law_id for article in articles})
        neighbors: Dict[str, Set[str]] = defaultdict(set)
        for article in articles:
            self._link(article, by_number, resolver, neighbors)
        return {point_id: sorted(targets) for point_id, targets in neighbors.items()}

    def _link(
        self,
        article: da.LawArticle,
        by_number: Dict[Tuple[str, int], da.LawArticle],
        resolver: LawNameResolver,
        neighbors: Dict[str, Set[str]],
    ) -> None:
        for span in CITATION_SPAN.finditer(article.text):
            law_id = resolver.resolve(span.group(2)) or article.law_id
            for number in ARTICLE_NUMBER.findall(span.group(1)):
                self._add(article, by_number.get((law_id, int(number))), neighbors)
        for number in ARTICLE_NUMBER.findall(article.text):
            self._add(article, by_number.get((article.law_id, int(number))), neighbors)

    @staticmethod
    def _add(
        source: da.LawArticle,
        target: Optional[da.LawArticle],
        neighbors: Dict[str, Set[str]],
    ) -> None:
        if target is None or target.point_id == source.point_id:
            return
        neighbors[source.point_id].add(target.point_id)
        neighbors[target.point_id].add(source.point_id)


class LegalGraphIndex:
    def __init__(self, db: "InMemoryGraphDB") -> None:
        self._db = db

    def __len__(self) -> int:
        return len(self._db.get_nodes_by_type(GRAPH_NODE_TYPE))

    def semantic_search(self, query_vector: Sequence[float], top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        records = self._db.find_similar_nodes(np.asarray(query_vector, dtype=np.float32), GRAPH_NODE_TYPE, top_k)
        return [(record, float(record.get("similarity", 0.0))) for record in records]

    def neighbors(self, point_id: str) -> List[str]:
        return list(dict.fromkeys(self._db.get_neighbors(point_id)))

    def get(self, point_id: str) -> Optional[Dict[str, Any]]:
        return self._db.get_node(point_id)

    def edge_count(self) -> int:
        return self._db.graph.number_of_edges()

    @staticmethod
    def save(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        GraphDBManager.save(path)

    @classmethod
    def load(cls, path: str) -> "LegalGraphIndex":
        GraphDBManager.load(path)
        return cls(GraphDBManager.get_db())

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(path)


class LegalGraphIndexBuilder:
    def __init__(
        self,
        loader: da.JsonLawCorpusLoader,
        citation_builder: CitationGraphBuilder,
        embedder: da.DenseEmbedder,
        knn_top_k: int,
    ) -> None:
        self._loader = loader
        self._citation_builder = citation_builder
        self._embedder = embedder
        self._knn_top_k = knn_top_k

    def build(self) -> LegalGraphIndex:
        """Build the graph store: law articles as embedded nodes, citation + KNN edges, then PageRank/community annotations."""
        articles = self._loader.load()
        print(f"🧩 [graph] loaded {len(articles)} law articles", flush=True)
        neighbors = self._citation_builder.build(articles)
        embeddings = self._embedder.encode_documents([article.text for article in articles])

        GraphDBManager.initialize()
        db = GraphDBManager.get_db()
        for article, embedding in zip(articles, embeddings):
            db.add_node(
                article.point_id,
                GRAPH_NODE_TYPE,
                {
                    "law_id": article.law_id,
                    "aid": article.aid,
                    "description": article.text,
                    "embedding": embedding,
                    "neighbors": neighbors.get(article.point_id, []),
                },
            )

        self._add_citation_edges(db, [article.point_id for article in articles], neighbors)
        if self._knn_top_k > 0:
            _feature_graph.run_knn(top_k=self._knn_top_k)
        for point_id, score in db.compute_pagerank().items():
            db.update_node(point_id, {"pagerank": float(score)})
        for point_id, community in db.detect_communities().items():
            db.update_node(point_id, {"communityId": int(community)})

        print(
            f"[graph] built {len(db.get_nodes_by_type(GRAPH_NODE_TYPE))} law nodes | "
            f"{db.graph.number_of_edges()} edges (citation + knn top-{self._knn_top_k})",
            flush=True,
        )
        return LegalGraphIndex(db)

    @staticmethod
    def _add_citation_edges(
        db: "InMemoryGraphDB",
        ids: Sequence[str],
        neighbors: Dict[str, List[str]],
    ) -> None:
        present = set(ids)
        for source in ids:
            for target in neighbors.get(source, []):
                if target in present:
                    db.add_edge(source, target, RELATION_CITES)


class AgenticGraphRetriever:
    def __init__(
        self,
        index: LegalGraphIndex,
        embedder: da.DenseEmbedder,
        assessor: da.RetrievalAssessor,
        seed_size: int,
        expand_top: int,
        neighbor_hops: int,
        max_iterations: int,
        max_evidence: int,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._assessor = assessor
        self._seed_size = seed_size
        self._expand_top = expand_top
        self._neighbor_hops = neighbor_hops
        self._max_iterations = max_iterations
        self._max_evidence = max_evidence

    def retrieve(
        self,
        case: da.LegalCase,
        decomposition: da.CaseQueryDecomposition,
    ) -> List[GraphLawReference]:
        """Seed by vector search, expand along graph neighbors, then let the LLM assessor request more rounds until sufficient."""
        collected: Dict[str, GraphLawReference] = {}
        queries = list(dict.fromkeys(decomposition.sub_queries + decomposition.keywords))
        for iteration in range(self._max_iterations):
            self._seed(queries, collected)
            self._expand(collected)
            ranked = self._rank(collected)
            print(
                f"[graph-law] case={case.case_id} iter={iteration + 1} "
                f"queries={len(queries)} collected={len(collected)}",
                flush=True,
            )
            if iteration + 1 >= self._max_iterations:
                break
            assessment = self._assessor.assess(case, ranked[: self._max_evidence])
            if assessment.sufficient:
                break
            queries = list(
                dict.fromkeys(assessment.additional_keywords + assessment.missing_information)
            )
            if not queries:
                break
        return self._rank(collected)[: self._max_evidence]

    def _seed(self, queries: Sequence[str], collected: Dict[str, GraphLawReference]) -> None:
        for query in queries:
            normalized = query.strip()
            if not normalized:
                continue
            query_vector = self._embedder.encode_query(normalized)
            for record, score in self._index.semantic_search(query_vector, self._seed_size):
                current = collected.get(record["id"])
                if current is None or score > current.score:
                    collected[record["id"]] = self._to_reference(record, score)

    def _expand(self, collected: Dict[str, GraphLawReference]) -> None:
        """Pull in graph neighbors of the current top nodes, up to `neighbor_hops` hops out."""
        for _ in range(self._neighbor_hops):
            top = self._rank(collected)[: self._expand_top]
            frontier: List[str] = []
            for reference in top:
                for neighbor_id in self._index.neighbors(reference.point_id):
                    if neighbor_id not in collected:
                        frontier.append(neighbor_id)
            frontier = list(dict.fromkeys(frontier))
            if not frontier:
                break
            for neighbor_id in frontier:
                if neighbor_id in collected:
                    continue
                node = self._index.get(neighbor_id)
                if node is not None:
                    collected[neighbor_id] = self._to_reference({"id": neighbor_id, **node}, 0.0)

    @staticmethod
    def _to_reference(record: Dict[str, Any], score: float) -> GraphLawReference:
        return GraphLawReference(
            point_id=record["id"],
            law_id=str(record.get("law_id", "")),
            aid=int(record.get("aid", 0)),
            text=str(record.get("description", "")),
            score=score,
            neighbors=tuple(record.get("neighbors", []) or []),
        )

    @staticmethod
    def _rank(collected: Dict[str, GraphLawReference]) -> List[GraphLawReference]:
        return sorted(collected.values(), key=lambda reference: reference.score, reverse=True)


class GraphRagPipeline:
    def __init__(
        self,
        decomposer: da.CaseQueryDecomposer,
        retriever: AgenticGraphRetriever,
        evidence_collector: da.CaseEvidenceCollector,
        predictor: da.ClassifierOutcomePredictor,
        max_law_evidence: int,
    ) -> None:
        self._decomposer = decomposer
        self._retriever = retriever
        self._evidence_collector = evidence_collector
        self._predictor = predictor
        self._max_law_evidence = max_law_evidence

    def run_case(self, case: da.LegalCase) -> da.SubmissionEntry:
        decomposition = self._decomposer.decompose(case)
        references = self._retriever.retrieve(case, decomposition)
        evidence = self._evidence_collector.collect(case, decomposition, references)
        reasoning = self._predictor.predict(case, references, evidence)
        return da.SubmissionEntry(
            case_id=case.case_id,
            case_evidence=[chunk.chunk_id for chunk in evidence],
            law_evidence=self._law_evidence(references),
            prediction=reasoning.prediction,
        )

    def _law_evidence(self, references: Sequence[GraphLawReference]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, int]] = set()
        for reference in references[: self._max_law_evidence]:
            key = (reference.law_id, reference.aid)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(reference.as_evidence())
        return evidence

    def run_dataset(
        self,
        cases: Sequence[da.LegalCase],
        checkpoint: Optional[Callable[[List[da.SubmissionEntry]], None]] = None,
    ) -> List[da.SubmissionEntry]:
        entries: List[da.SubmissionEntry] = []
        for position, case in enumerate(cases, start=1):
            print(f"[case {position}/{len(cases)}] {case.case_id}", flush=True)
            try:
                entries.append(self.run_case(case))
            except Exception as error:
                print(
                    f"⚠️ [case {case.case_id}] FAILED ({error}) -> fallback "
                    f"'{da.DEFAULT_PREDICTION}'",
                    flush=True,
                )
                entries.append(
                    da.SubmissionEntry(
                        case_id=case.case_id,
                        case_evidence=[],
                        law_evidence=[],
                        prediction=da.DEFAULT_PREDICTION,
                    )
                )
            if checkpoint is not None:
                checkpoint(entries)
        return entries


class GraphPipelineFactory:
    @staticmethod
    def create(
        settings: da.Settings,
        index: LegalGraphIndex,
        embedder: da.DenseEmbedder,
    ) -> GraphRagPipeline:
        model = da.LanguageModelFactory.create(settings)
        parser = da.JsonResponseParser()
        decomposer = da.CaseQueryDecomposer(model, parser)
        assessor = da.RetrievalAssessor(model, parser)
        retriever = AgenticGraphRetriever(
            index=index,
            embedder=embedder,
            assessor=assessor,
            seed_size=settings.law_search_limit,
            expand_top=EXPAND_TOP,
            neighbor_hops=NEIGHBOR_HOPS,
            max_iterations=settings.max_retrieval_iterations,
            max_evidence=settings.max_law_evidence,
        )
        case_client = da.CaseContentClient(
            api_url=settings.case_api_url,
            api_token=settings.case_api_token,
            rate_limiter=da.RateLimiter(settings.api_min_interval),
            max_retries=settings.api_max_retries,
            rate_limit_delay=settings.api_rate_limit_delay,
            max_rate_limit_retries=settings.api_max_rate_limit_retries,
        )
        extractor = da.CaseKeywordExtractor(model, parser)
        evidence_collector = da.CaseEvidenceCollector(
            client=case_client,
            extractor=extractor,
            probe_queries=da.JUDGMENT_PROBE_QUERIES,
            max_calls=settings.max_case_queries,
        )
        predictor = da.ClassifierOutcomePredictor(
            model_id=settings.outcome_model_id,
            device=settings.device,
            max_length=settings.outcome_max_length,
        )
        return GraphRagPipeline(
            decomposer=decomposer,
            retriever=retriever,
            evidence_collector=evidence_collector,
            predictor=predictor,
            max_law_evidence=settings.max_law_evidence,
        )


def _load_existing_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    return [record for record in data if isinstance(record, dict)] if isinstance(data, list) else []


def _entry_from_dict(record: Dict[str, Any]) -> da.SubmissionEntry:
    return da.SubmissionEntry(
        case_id=str(record.get("case_id", "")),
        case_evidence=list(record.get("case_evidence", []) or []),
        law_evidence=list(record.get("law_evidence", []) or []),
        prediction=str(record.get("prediction", da.DEFAULT_PREDICTION)),
    )


def _next_output_path(base: Path) -> Path:
    index = 2
    while True:
        candidate = base.with_name(f"{base.stem}-{index}{base.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _resolve_resume(
    base: Path,
    case_ids: Sequence[str],
) -> Tuple[Path, Dict[str, da.SubmissionEntry], Set[str]]:
    existing = _load_existing_entries(base)
    done = {str(record.get("case_id")) for record in existing}
    if existing and done.issuperset(case_ids):
        return _next_output_path(base), {}, set()
    entries = {str(record.get("case_id")): _entry_from_dict(record) for record in existing}
    return base, entries, set(entries)


class Application:
    def __init__(self, settings: da.Settings, graph_path: str) -> None:
        self._settings = settings
        self._graph_path = graph_path
        self._embedder = da.EmbedderFactory.create_dense(settings)

    def build_graph(self, recreate: bool) -> LegalGraphIndex:
        if not recreate and LegalGraphIndex.exists(self._graph_path):
            print(f"🧩 [graph] found existing store -> skip stage 1: {self._graph_path}", flush=True)
            return LegalGraphIndex.load(self._graph_path)
        print("🧩 [graph] stage 1: building LegalGraphRAG in-memory graph DB (Halong embeddings + citation/knn)", flush=True)
        loader = da.JsonLawCorpusLoader(self._settings.corpus_path)
        index = LegalGraphIndexBuilder(
            loader,
            CitationGraphBuilder(),
            self._embedder,
            knn_top_k=KNN_TOP_K,
        ).build()
        LegalGraphIndex.save(self._graph_path)
        print(f"🧩 [graph] saved store: {self._graph_path}", flush=True)
        return index

    def index(self, recreate: bool) -> None:
        self.build_graph(recreate=recreate)

    def infer(self, limit: Optional[int], recreate: bool) -> None:
        index = self.build_graph(recreate=recreate)
        pipeline = GraphPipelineFactory.create(self._settings, index, self._embedder)
        cases = da.CaseDatasetLoader(self._settings.input_path).load(limit=limit)
        case_ids = [case.case_id for case in cases]
        base = Path(self._settings.output_path)
        output_path, entries, done = _resolve_resume(base, case_ids)
        if output_path != base:
            print(f"🔁 [resume] {base} đã hoàn tất -> ghi file mới: {output_path}", flush=True)
        elif done:
            print(f"🔁 [resume] tiếp tục: đã có {len(done)}/{len(cases)} vụ án trong {output_path}", flush=True)
        writer = da.SubmissionWriter(output_path)
        bar = tqdm(
            total=len(cases),
            initial=len(done),
            desc="⚖️  LegalGraph",
            colour="green",
            unit="case",
            dynamic_ncols=True,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        for case in cases:
            if case.case_id in done:
                continue
            entry = pipeline.run_case(case)
            entries[case.case_id] = entry
            writer.write([entries[cid] for cid in case_ids if cid in entries], quiet=True)
            bar.update(1)
            bar.set_postfix_str(f"📚 {case.case_id} → {entry.prediction}")
        bar.close()
        writer.write([entries[cid] for cid in case_ids if cid in entries])


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALQAC LegalGraph RAG (LegalGraphRAG in-memory graph DB + Halong)")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--graph-db-path", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--llm-provider", type=str, default=None)
    parser.add_argument("--vllm-api-base", type=str, default=None)
    parser.add_argument("--vllm-model-name", type=str, default=None)
    parser.add_argument("--vllm-api-key", type=str, default=None)
    return parser.parse_args(list(argv))


def _build_settings(args: argparse.Namespace) -> da.Settings:
    overrides: Dict[str, Any] = {
        "corpus_path": Path(args.corpus_path) if args.corpus_path else DATA_DIR / "corpus_law_pub.json",
        "input_path": Path(args.input_path) if args.input_path else DATA_DIR / "ALQAC2026_public_test.json",
        "output_path": Path(args.output_path) if args.output_path else Path("submission_legal_graph.json"),
        "llm_provider": args.llm_provider or os.environ.get("LLM_PROVIDER", "vllm"),
    }
    if args.device:
        overrides["device"] = args.device
    if args.vllm_api_base:
        overrides["vllm_api_base"] = args.vllm_api_base
    if args.vllm_model_name:
        overrides["vllm_model_name"] = args.vllm_model_name
    if args.vllm_api_key:
        overrides["vllm_api_key"] = args.vllm_api_key
    return da.Settings.from_environment(overrides)


def _resolve_graph_path(graph_db_path: Optional[str], collection_name: Optional[str]) -> str:
    graph_db = Path(graph_db_path) if graph_db_path else DEFAULT_GRAPH_DB_PATH
    collection = Path(collection_name) if collection_name else Path(DEFAULT_COLLECTION_NAME)
    store = collection if collection.is_absolute() else graph_db / collection
    store.parent.mkdir(parents=True, exist_ok=True)
    graph_db.mkdir(parents=True, exist_ok=True)
    return str(store)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    settings = _build_settings(args)
    graph_path = _resolve_graph_path(args.graph_db_path, args.collection_name)
    application = Application(settings, graph_path)
    if args.index:
        application.index(recreate=True)
        return 0
    application.infer(limit=args.limit, recreate=args.recreate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
