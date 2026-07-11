from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

import deep_agents as da

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "legal_graph_store.pkl"

CITATION_SPAN = re.compile(
    r"((?:(?:kho[aả]n\s+\d+\s+)?[Đđ]i[eề]u\s+\d+[\s,;và]*)+)\s*(?:c[uủ]a\s+)?"
    r"((?:B[ộo]\s+lu[ậa]t|Lu[ậa]t|Ngh[ịi]\s+quy[ếe]t|Ngh[ịi]\s+đ[ịi]nh|Ph[áa]p\s+l[ệe]nh)[^;.\n]{0,50})"
)
ARTICLE_NUMBER = re.compile(r"[Đđ]i[eề]u\s+(\d+)")

SEED_SIZE: int = 10
EXPAND_TOP: int = 5
NEIGHBOR_HOPS: int = 2


@dataclass
class GraphNode:
    point_id: str
    law_id: str
    aid: int
    text: str
    neighbors: List[str] = field(default_factory=list)


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


class InMemoryVectorGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self._ids)

    def add_nodes(self, nodes: Sequence[GraphNode], embeddings: Sequence[Sequence[float]]) -> None:
        self._nodes = {node.point_id: node for node in nodes}
        self._ids = [node.point_id for node in nodes]
        self._matrix = np.asarray(embeddings, dtype=np.float32)

    def semantic_search(self, query_vector: Sequence[float], top_k: int) -> List[Tuple[GraphNode, float]]:
        if self._matrix is None or not self._ids:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        scores = self._matrix @ query
        limit = min(top_k, len(scores))
        if limit <= 0:
            return []
        pivot = np.argpartition(-scores, limit - 1)[:limit]
        order = pivot[np.argsort(-scores[pivot])]
        return [(self._nodes[self._ids[index]], float(scores[index])) for index in order]

    def neighbors(self, point_id: str) -> List[str]:
        node = self._nodes.get(point_id)
        return list(node.neighbors) if node else []

    def get(self, point_id: str) -> Optional[GraphNode]:
        return self._nodes.get(point_id)

    def edge_count(self) -> int:
        return sum(len(node.neighbors) for node in self._nodes.values())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self._ids,
            "matrix": self._matrix,
            "nodes": {
                point_id: {
                    "law_id": node.law_id,
                    "aid": node.aid,
                    "text": node.text,
                    "neighbors": node.neighbors,
                }
                for point_id, node in self._nodes.items()
            },
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "InMemoryVectorGraph":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        graph = cls()
        graph._ids = list(payload["ids"])
        graph._matrix = np.asarray(payload["matrix"], dtype=np.float32)
        graph._nodes = {
            point_id: GraphNode(
                point_id=point_id,
                law_id=data["law_id"],
                aid=int(data["aid"]),
                text=data["text"],
                neighbors=list(data["neighbors"]),
            )
            for point_id, data in payload["nodes"].items()
        }
        return graph

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(path)


class GraphBuilder:
    def __init__(
        self,
        loader: da.JsonLawCorpusLoader,
        citation_builder: CitationGraphBuilder,
        embedder: da.DenseEmbedder,
    ) -> None:
        self._loader = loader
        self._citation_builder = citation_builder
        self._embedder = embedder

    def build(self) -> InMemoryVectorGraph:
        articles = self._loader.load()
        print(f"[graph] loaded {len(articles)} law articles", flush=True)
        neighbors = self._citation_builder.build(articles)
        nodes = [
            GraphNode(
                point_id=article.point_id,
                law_id=article.law_id,
                aid=article.aid,
                text=article.text,
                neighbors=neighbors.get(article.point_id, []),
            )
            for article in articles
        ]
        embeddings = self._embedder.encode_documents([article.text for article in articles])
        graph = InMemoryVectorGraph()
        graph.add_nodes(nodes, embeddings)
        print(f"[graph] built {len(graph)} nodes | {graph.edge_count()} directed citation edges", flush=True)
        return graph


class AgenticGraphRetriever:
    def __init__(
        self,
        graph: InMemoryVectorGraph,
        embedder: da.DenseEmbedder,
        assessor: da.RetrievalAssessor,
        seed_size: int,
        expand_top: int,
        neighbor_hops: int,
        max_iterations: int,
        max_evidence: int,
    ) -> None:
        self._graph = graph
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
            for node, score in self._graph.semantic_search(query_vector, self._seed_size):
                current = collected.get(node.point_id)
                if current is None or score > current.score:
                    collected[node.point_id] = self._to_reference(node, score)

    def _expand(self, collected: Dict[str, GraphLawReference]) -> None:
        for _ in range(self._neighbor_hops):
            top = self._rank(collected)[: self._expand_top]
            frontier: List[str] = []
            for reference in top:
                for neighbor_id in reference.neighbors:
                    if neighbor_id not in collected:
                        frontier.append(neighbor_id)
            frontier = list(dict.fromkeys(frontier))
            if not frontier:
                break
            for neighbor_id in frontier:
                node = self._graph.get(neighbor_id)
                if node is not None and node.point_id not in collected:
                    collected[node.point_id] = self._to_reference(node, 0.0)

    @staticmethod
    def _to_reference(node: GraphNode, score: float) -> GraphLawReference:
        return GraphLawReference(
            point_id=node.point_id,
            law_id=node.law_id,
            aid=node.aid,
            text=node.text,
            score=score,
            neighbors=tuple(node.neighbors),
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

    def run_dataset(self, cases: Sequence[da.LegalCase]) -> List[da.SubmissionEntry]:
        entries: List[da.SubmissionEntry] = []
        for position, case in enumerate(cases, start=1):
            print(f"[case {position}/{len(cases)}] {case.case_id}", flush=True)
            entries.append(self.run_case(case))
        return entries


class GraphPipelineFactory:
    @staticmethod
    def create(
        settings: da.Settings,
        graph: InMemoryVectorGraph,
        embedder: da.DenseEmbedder,
    ) -> GraphRagPipeline:
        model = da.LanguageModelFactory.create(settings)
        parser = da.JsonResponseParser()
        decomposer = da.CaseQueryDecomposer(model, parser)
        assessor = da.RetrievalAssessor(model, parser)
        retriever = AgenticGraphRetriever(
            graph=graph,
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


class Application:
    def __init__(self, settings: da.Settings, graph_path: str) -> None:
        self._settings = settings
        self._graph_path = graph_path
        self._embedder = da.EmbedderFactory.create_dense(settings)

    def build_graph(self, recreate: bool) -> InMemoryVectorGraph:
        if not recreate and InMemoryVectorGraph.exists(self._graph_path):
            print(f"[graph] found existing store -> skip stage 1: {self._graph_path}", flush=True)
            return InMemoryVectorGraph.load(self._graph_path)
        print("[graph] stage 1: building citation graph + Halong embeddings", flush=True)
        loader = da.JsonLawCorpusLoader(self._settings.corpus_path)
        graph = GraphBuilder(loader, CitationGraphBuilder(), self._embedder).build()
        graph.save(self._graph_path)
        print(f"[graph] saved store: {self._graph_path}", flush=True)
        return graph

    def index(self, recreate: bool) -> None:
        self.build_graph(recreate=recreate)

    def infer(self, limit: Optional[int], recreate: bool) -> None:
        graph = self.build_graph(recreate=recreate)
        pipeline = GraphPipelineFactory.create(self._settings, graph, self._embedder)
        cases = da.CaseDatasetLoader(self._settings.input_path).load(limit=limit)
        entries = pipeline.run_dataset(cases)
        da.SubmissionWriter(self._settings.output_path).write(entries)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALQAC LegalGraph RAG (in-memory graph + Halong, 2-stage)")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--corpus-path", type=str, default=None)
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--graph-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(list(argv))


def _build_settings(args: argparse.Namespace) -> da.Settings:
    overrides: Dict[str, Any] = {
        "corpus_path": Path(args.corpus_path) if args.corpus_path else DATA_DIR / "corpus_law_pub.json",
        "input_path": Path(args.input_path) if args.input_path else DATA_DIR / "ALQAC2026_public_test.json",
        "output_path": Path(args.output_path) if args.output_path else Path("submission_legal_graph.json"),
    }
    if args.device:
        overrides["device"] = args.device
    return da.Settings.from_environment(overrides)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    settings = _build_settings(args)
    graph_path = args.graph_path or str(DEFAULT_GRAPH_PATH)
    application = Application(settings, graph_path)
    if args.index:
        application.index(recreate=True)
        return 0
    application.infer(limit=args.limit, recreate=args.recreate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
