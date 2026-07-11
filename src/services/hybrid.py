from __future__ import annotations

from typing import Dict, List, Sequence, Set

import ALQAC.src.services.deep_searcher as ds
import ALQAC.src.services.legal_graph as lg

SUBQUERIES_PER_ROUND: int = 4


class HybridEvidenceCollector:
    def __init__(
        self,
        source: lg.CaseSegmentSource,
        planner: ds.SubQueryPlanner,
        probe_queries: Sequence[str],
        max_calls: int,
    ) -> None:
        self._source = source
        self._planner = planner
        self._probe_queries = list(probe_queries)
        self._max_calls = max_calls

    def collect(self, case: lg.LegalCase, features: lg.CaseFeatures) -> List[lg.RetrievedChunk]:
        if not self._source.enabled:
            return []
        collected: Dict[str, lg.RetrievedChunk] = {}
        issued: Set[str] = set()
        queries = self._seed_queries(case, features)
        while queries and self._within_budget(issued, collected):
            added = self._run_round(queries, case, issued, collected)
            if added == 0:
                break
            gap = self._planner.gap_queries(case, list(collected.values()))
            queries = [query for query in gap if query.strip() and query.strip() not in issued]
        return list(collected.values())

    def _within_budget(self, issued: Set[str], collected: Dict[str, lg.RetrievedChunk]) -> bool:
        if len(issued) >= self._max_calls:
            return False
        return len(issued) < 2 * max(len(collected), ds.PENALTY_SAFE_MIN_SEGMENTS)

    def _seed_queries(self, case: lg.LegalCase, features: lg.CaseFeatures) -> List[str]:
        ordered: List[str] = [case.case_query]
        ordered.extend(self._probe_queries)
        ordered.extend(features.plaintiff_claims)
        ordered.extend(features.disputed_issues)
        ordered.extend(features.defendant_defense)
        ordered.extend(self._planner.initial_queries(case))
        unique: List[str] = []
        seen: Set[str] = set()
        for query in ordered:
            normalized = query.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _run_round(
        self,
        queries: Sequence[str],
        case: lg.LegalCase,
        issued: Set[str],
        collected: Dict[str, lg.RetrievedChunk],
    ) -> int:
        added = 0
        for query in queries:
            if not self._within_budget(issued, collected):
                break
            normalized = query.strip()
            if not normalized or normalized in issued:
                continue
            issued.add(normalized)
            before = len(collected)
            for chunk in self._source.search(normalized, case.case_id):
                if chunk.chunk_id not in collected:
                    collected[chunk.chunk_id] = chunk
            added += len(collected) - before
        return added


class HybridPipeline:
    def __init__(
        self,
        feature_extractor: lg.CaseFeatureExtractor,
        evidence_collector: HybridEvidenceCollector,
        law_retriever: lg.LawEvidenceRetriever,
        judge: lg.OutcomeJudge,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._evidence_collector = evidence_collector
        self._law_retriever = law_retriever
        self._judge = judge

    def run_case(self, case: lg.LegalCase) -> lg.SubmissionRecord:
        features = self._feature_extractor.extract(case)
        evidence = self._evidence_collector.collect(case, features)
        laws = self._law_retriever.retrieve(case, features, evidence)
        prediction = self._judge.judge(case, features, evidence, laws)
        return lg.SubmissionRecord(
            case_id=case.case_id,
            prediction=prediction,
            case_evidence=[chunk.chunk_id for chunk in evidence],
            law_evidence=laws,
        )

    def run(self, cases: Sequence[lg.LegalCase]) -> List[lg.SubmissionRecord]:
        records: List[lg.SubmissionRecord] = []
        for position, case in enumerate(cases, start=1):
            print(f"[{position}/{len(cases)}] {case.case_id}", flush=True)
            records.append(self.run_case(case))
        return records


class HybridPipelineBuilder:
    @staticmethod
    def build(config: lg.AppConfig, articles: Sequence[lg.LawArticle]) -> HybridPipeline:
        model = lg.PipelineBuilder._make_model(config, config.llm_temperature)
        cache = lg.ArtifactCache(config.cache_dir, config.read_cache, config.write_cache)
        fingerprint = lg.fingerprint_file(config.law_path)
        texts = [article.content for article in articles]
        index = cache.load_or_build(
            f"bm25|{lg.CACHE_VERSION}|{fingerprint}",
            lambda: lg.Bm25Index(texts),
        )
        graph = cache.load_or_build(
            f"graph|{lg.CACHE_VERSION}|{fingerprint}",
            lambda: lg.LegalCitationGraph(articles),
        )
        feature_extractor = lg.CaseFeatureExtractor(model=model)
        planner = ds.SubQueryPlanner(model=model, queries_per_round=SUBQUERIES_PER_ROUND)
        evidence_collector = HybridEvidenceCollector(
            source=lg.CaseSourceFactory.create(config),
            planner=planner,
            probe_queries=lg.JUDGMENT_PROBE_QUERIES,
            max_calls=config.max_case_calls,
        )
        graph_fallback = lg.GraphLawRetriever(
            model=model,
            graph=graph,
            index=index,
            seed_size=config.seed_size,
            neighbor_limit=config.neighbor_limit,
            shortlist_size=config.law_shortlist_size,
            max_results=config.max_law_evidence,
        )
        law_retriever = lg.CitationLawExtractor(
            resolver=lg.LawNameResolver(corpus_law_ids={article.law_id for article in articles}),
            index=lg.ArticleNumberIndex(articles),
            max_results=config.max_law_evidence,
            fallback=graph_fallback,
        )
        outcome_model = model
        if config.outcome_samples > 1:
            outcome_model = lg.PipelineBuilder._make_model(config, config.outcome_temperature)
        judge = lg.LlmOutcomeJudge(model=outcome_model, graph=graph, samples=config.outcome_samples)
        return HybridPipeline(feature_extractor, evidence_collector, law_retriever, judge)


def main() -> None:
    config = lg.parse_arguments()
    cases = lg.CaseRepository.load(config.test_path)
    articles = lg.LawRepository.load(config.law_path)
    print(f"Đã nạp {len(cases)} vụ án và {len(articles)} điều luật (HYBRID: deep sub-query + legal feature/graph).", flush=True)
    pipeline = HybridPipelineBuilder.build(config, articles)
    records = pipeline.run(cases)
    lg.SubmissionWriter.write(config.output_path, records)
    print(f"Đã ghi {len(records)} kết quả vào {config.output_path}.", flush=True)


if __name__ == "__main__":
    main()
