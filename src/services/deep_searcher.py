from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


PREDICTION_LABELS: Tuple[str, ...] = ("A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN")
DEFAULT_PREDICTION: str = "PARTIAL_A_WIN"
TOKEN_ENV_NAMES: Tuple[str, ...] = ("ALQAC_TOKEN", "ALQAC_API_KEY", "X_API_KEY")
CACHE_VERSION: str = "v1"

JUDGMENT_PROBE_QUERIES: Tuple[str, ...] = (
    "Quyết định của Tòa án tuyên xử",
    "Chấp nhận một phần yêu cầu khởi kiện của nguyên đơn",
    "Không chấp nhận một phần yêu cầu khởi kiện",
    "Bác một phần yêu cầu của nguyên đơn",
    "Chấp nhận toàn bộ yêu cầu khởi kiện của nguyên đơn",
    "Không chấp nhận toàn bộ yêu cầu khởi kiện",
    "Căn cứ áp dụng các Điều khoản của Bộ luật Dân sự",
    "Áp dụng Bộ luật Tố tụng dân sự Điều 157 Điều 158",
    "án phí dân sự sơ thẩm Nghị quyết 326 lệ phí Tòa án",
    "Buộc bị đơn phải bồi thường cho nguyên đơn số tiền",
    "Nhận định của Hội đồng xét xử về nội dung tranh chấp",
    "yêu cầu của nguyên đơn về bồi thường thiệt hại",
    "lời khai trình bày của nguyên đơn",
    "ý kiến trình bày của bị đơn",
    "chứng cứ tài liệu có trong hồ sơ vụ án",
    "kết quả thẩm định giá tài sản",
    "ý kiến của Viện kiểm sát nhân dân",
)

PENALTY_SAFE_MIN_SEGMENTS: int = 3

CacheableT = TypeVar("CacheableT")


@dataclass(frozen=True)
class AppConfig:
    test_path: str
    law_path: str
    output_path: str
    api_url: str
    api_key: Optional[str]
    request_interval: float
    request_timeout: float
    max_retries: int
    rate_limit_delay: float
    max_rate_limit_retries: int
    max_case_calls: int
    subqueries_per_round: int
    law_shortlist_size: int
    max_law_evidence: int
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_temperature: float
    llm_max_retries: int
    llm_min_interval: float
    llm_retry_delay: float
    fallback_model: str
    fallback_base_url: str
    use_fallback: bool
    outcome_samples: int
    outcome_temperature: float
    cache_dir: str
    read_cache: bool
    write_cache: bool


@dataclass(frozen=True)
class LegalCase:
    case_id: str
    case_query: str
    plaintiff_role: str
    defendant_role: str


@dataclass(frozen=True)
class LawArticle:
    law_id: str
    aid: int
    content: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True)
class LawReference:
    law_id: str
    aid: int


@dataclass
class SubmissionRecord:
    case_id: str
    prediction: str
    case_evidence: List[str] = field(default_factory=list)
    law_evidence: List[LawReference] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "prediction": self.prediction,
            "case_evidence": list(self.case_evidence),
            "law_evidence": [{"law_id": ref.law_id, "aid": ref.aid} for ref in self.law_evidence],
        }


class JsonExtractor:
    _FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

    @classmethod
    def extract_object(cls, raw: str) -> Optional[Dict[str, object]]:
        value = cls._extract(raw, "{", "}")
        return value if isinstance(value, dict) else None

    @classmethod
    def extract_array(cls, raw: str) -> Optional[List[object]]:
        value = cls._extract(raw, "[", "]")
        return value if isinstance(value, list) else None

    @classmethod
    def _extract(cls, raw: str, opener: str, closer: str) -> Optional[object]:
        candidates: List[str] = []
        fenced = cls._FENCE_PATTERN.search(raw)
        if fenced:
            candidates.append(fenced.group(1))
        candidates.append(raw)
        for candidate in candidates:
            parsed = cls._scan(candidate, opener, closer)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _scan(text: str, opener: str, closer: str) -> Optional[object]:
        decoder = json.JSONDecoder()
        start = text.find(opener)
        while start != -1:
            try:
                value, _ = decoder.raw_decode(text[start:])
                return value
            except json.JSONDecodeError:
                start = text.find(opener, start + 1)
        _ = closer
        return None


class LanguageModel(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class LiteLlmLanguageModel(LanguageModel):
    _last_call_at: float = 0.0

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_retries: int,
        min_interval: float = 0.0,
        retry_delay: float = 5.0,
    ) -> None:
        self._model_name = model_name
        self._max_retries = max_retries
        self._min_interval = min_interval
        self._retry_delay = retry_delay
        self._client = self._build_client(model_name, base_url, api_key, temperature)

    @staticmethod
    def _build_client(model_name: str, base_url: str, api_key: str, temperature: float):
        from langchain_litellm import ChatLiteLLMRouter
        from litellm import Router

        if model_name.startswith("azure/"):
            litellm_params = {
                "model": model_name,
                "base_model": "gpt-4o",
                "api_key": os.environ.get("AZURE_API_KEY"),
                "api_version": os.environ.get("AZURE_API_VERSION"),
                "api_base": os.environ.get("AZURE_API_BASE"),
                "timeout": 60 * 60,
            }
        else:
            litellm_params = {
                "model": f"openai/{model_name}",
                "api_base": base_url,
                "api_key": api_key,
                "timeout": 60 * 60,
            }
        router = Router(model_list=[{"model_name": model_name, "litellm_params": litellm_params}])
        return ChatLiteLLMRouter(router=router, model_name=model_name, temperature=temperature)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            self._respect_rate_limit()
            try:
                response = self._client.invoke(prompt)
                return self._to_text(getattr(response, "content", response))
            except Exception as error:
                last_error = error
                print(
                    f"LLM '{self._model_name}' lỗi (lần {attempt + 1}/{self._max_retries}): {error} "
                    f"-> chờ {self._retry_delay * (attempt + 1):.0f}s rồi thử lại.",
                    flush=True,
                )
                time.sleep(self._retry_delay * (attempt + 1))
        raise RuntimeError(f"LLM generation failed after {self._max_retries} attempts: {last_error}")

    def _respect_rate_limit(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - LiteLlmLanguageModel._last_call_at
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        LiteLlmLanguageModel._last_call_at = time.monotonic()

    @staticmethod
    def _to_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts)
        return str(content)


class FallbackLanguageModel(LanguageModel):
    def __init__(self, primary: LanguageModel, fallback: LanguageModel) -> None:
        self._primary = primary
        self._fallback = fallback

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._primary.generate(system_prompt, user_prompt)
        except Exception as error:
            print(f"LLM chính thất bại ({error}) -> chuyển sang model dự phòng.", flush=True)
            return self._fallback.generate(system_prompt, user_prompt)


class CaseSegmentSource(ABC):
    @abstractmethod
    def search(self, query: str, case_id: str) -> List[RetrievedChunk]:
        raise NotImplementedError

    @property
    @abstractmethod
    def enabled(self) -> bool:
        raise NotImplementedError


class NullCaseSegmentSource(CaseSegmentSource):
    def search(self, query: str, case_id: str) -> List[RetrievedChunk]:
        _ = (query, case_id)
        return []

    @property
    def enabled(self) -> bool:
        return False


class HttpCaseSegmentSource(CaseSegmentSource):
    def __init__(
        self,
        api_url: str,
        api_key: str,
        request_interval: float,
        request_timeout: float,
        max_retries: int,
        rate_limit_delay: float,
        max_rate_limit_retries: int,
    ) -> None:
        import requests

        self._requests = requests
        self._api_url = api_url
        self._api_key = api_key
        self._interval = request_interval
        self._timeout = request_timeout
        self._max_retries = max_retries
        self._rate_limit_delay = rate_limit_delay
        self._max_rate_limit_retries = max_rate_limit_retries
        self._last_request_at = time.monotonic()

    @property
    def enabled(self) -> bool:
        return True

    def search(self, query: str, case_id: str) -> List[RetrievedChunk]:
        payload = {"query": query, "case_id": case_id}
        headers = {"X-API-Key": self._api_key, "Content-Type": "application/json"}
        error_attempts = 0
        rate_limit_hits = 0
        while error_attempts <= self._max_retries:
            self._respect_rate_limit()
            try:
                response = self._requests.post(
                    self._api_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
            except Exception:
                error_attempts += 1
                time.sleep(self._interval)
                continue
            if response.status_code == 429:
                rate_limit_hits += 1
                if rate_limit_hits > self._max_rate_limit_retries:
                    return []
                print(
                    f"Bị rate limit (429) -> chờ {self._rate_limit_delay:.0f}s rồi thử lại "
                    f"({rate_limit_hits}/{self._max_rate_limit_retries}).",
                    flush=True,
                )
                time.sleep(self._rate_limit_delay)
                continue
            if response.status_code != 200:
                return []
            return self._parse(response.json())
        return []

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _parse(body: Dict[str, object]) -> List[RetrievedChunk]:
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
            text = hit.get("text") if isinstance(hit.get("text"), str) else ""
            score = float(hit.get("score", 0.0)) if isinstance(hit.get("score"), (int, float)) else 0.0
            chunks.append(RetrievedChunk(chunk_id=chunk_id, text=text, score=score))
        return chunks


class Bm25Index:
    _TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._tokenized = [self._tokenize(document) for document in documents]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized]
        self._avg_length = (sum(self._doc_lengths) / len(self._doc_lengths)) if self._doc_lengths else 0.0
        self._term_frequencies = [Counter(tokens) for tokens in self._tokenized]
        self._inverse_document_frequency = self._compute_idf()

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return cls._TOKEN_PATTERN.findall(text.lower())

    def _compute_idf(self) -> Dict[str, float]:
        document_frequency: Counter = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        total_documents = len(self._tokenized)
        return {
            term: math.log(1.0 + (total_documents - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        query_terms = self._tokenize(query)
        scores: List[Tuple[int, float]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            scores.append((index, self._score(query_terms, index, frequencies)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def _score(self, query_terms: Sequence[str], index: int, frequencies: Counter) -> float:
        length = self._doc_lengths[index]
        denominator_base = self._k1 * (1 - self._b + self._b * (length / self._avg_length if self._avg_length else 1.0))
        total = 0.0
        for term in query_terms:
            if term not in frequencies:
                continue
            term_frequency = frequencies[term]
            idf = self._inverse_document_frequency.get(term, 0.0)
            total += idf * (term_frequency * (self._k1 + 1)) / (term_frequency + denominator_base)
        return total


class SubQueryPlanner:
    _SYSTEM = (
        "Bạn là nghiên cứu viên pháp lý thực hiện deep research trên bản án. Mục tiêu là thu thập đủ bằng chứng "
        "để xác định (a) bên thắng kiện và mức độ thắng, (b) các điều luật Tòa viện dẫn. Bạn tạo câu truy vấn "
        "tìm kiếm tiếng Việt và tự đánh giá khi nào đã đủ thông tin."
    )

    def __init__(self, model: LanguageModel, queries_per_round: int) -> None:
        self._model = model
        self._queries_per_round = queries_per_round

    def initial_queries(self, case: LegalCase) -> List[str]:
        user_prompt = (
            f"Mô tả vụ án:\n{case.case_query}\n\n"
            "Hãy PHÂN RÃ vụ án thành các câu truy vấn NGẮN (mỗi câu 3-8 từ), mỗi câu nắm MỘT pattern chính để "
            "tìm đúng đoạn bản án: loại tranh chấp; yêu cầu cụ thể của nguyên đơn; đối tượng tranh chấp "
            "(đất/nhà/tiền/di sản); số tiền hoặc diện tích; hành vi tranh chấp; phần Quyết định của Tòa; "
            "đoạn 'Căn cứ vào Điều... Bộ luật'. Câu càng ngắn và đúng trọng tâm càng tốt, không viết cả câu dài.\n"
            f"Tạo {self._queries_per_round + 4} câu truy vấn ngắn.\n"
            'Chỉ trả về JSON dạng: {"queries": ["...", "..."]}'
        )
        return self._parse_queries(self._model.generate(self._SYSTEM, user_prompt), fallback=[case.case_query])

    def gap_queries(self, case: LegalCase, collected: Sequence[RetrievedChunk]) -> List[str]:
        evidence = "\n".join(f"- {chunk.text[:220]}" for chunk in collected) or "(chưa có bằng chứng)"
        user_prompt = (
            f"Mô tả vụ án:\n{case.case_query}\n\n"
            f"Bằng chứng đã truy xuất được cho tới lúc này:\n{evidence}\n\n"
            "Hãy SUY LUẬN như deep research: với bằng chứng hiện có, đã ĐỦ để kết luận chắc chắn "
            "(a) bên thắng và mức độ thắng (toàn bộ / một phần trên 50% / một phần không quá 50% / thua) và "
            "(b) các điều luật được viện dẫn hay CHƯA?\n"
            "- Nếu CHƯA đủ (còn thiếu phần Quyết định, đoạn 'Căn cứ vào Điều', yêu cầu bị bác...): đặt "
            f"need_more=true và tạo tối đa {self._queries_per_round} câu truy vấn để gọi API tìm tiếp.\n"
            "- Nếu ĐÃ đủ để quyết định: đặt need_more=false và để queries rỗng (DỪNG gọi API).\n"
            'Chỉ trả về JSON dạng: {"need_more": true, "queries": ["..."]}'
        )
        payload = JsonExtractor.extract_object(self._model.generate(self._SYSTEM, user_prompt))
        if not payload or not bool(payload.get("need_more")):
            return []
        queries = payload.get("queries")
        if not isinstance(queries, list):
            return []
        return [str(item).strip() for item in queries if str(item).strip()]

    @staticmethod
    def _parse_queries(raw: str, fallback: List[str]) -> List[str]:
        payload = JsonExtractor.extract_object(raw)
        if not payload or not isinstance(payload.get("queries"), list):
            return fallback
        queries = [str(item).strip() for item in payload["queries"] if str(item).strip()]
        return queries if queries else fallback


class CaseEvidenceCollector:
    def __init__(
        self,
        source: CaseSegmentSource,
        planner: SubQueryPlanner,
        probe_queries: Sequence[str],
        max_calls: int,
    ) -> None:
        self._source = source
        self._planner = planner
        self._probe_queries = list(probe_queries)
        self._max_calls = max_calls

    def collect(self, case: LegalCase) -> List[RetrievedChunk]:
        if not self._source.enabled:
            return []
        collected: Dict[str, RetrievedChunk] = {}
        issued: Set[str] = set()
        queries = self._seed_queries(case)
        while queries and self._within_budget(issued, collected):
            added = self._run_round(queries, case, issued, collected)
            if added == 0:
                break
            gap = self._planner.gap_queries(case, list(collected.values()))
            queries = [query for query in gap if query.strip() and query.strip() not in issued]
        return list(collected.values())

    def _within_budget(self, issued: Set[str], collected: Dict[str, RetrievedChunk]) -> bool:
        if len(issued) >= self._max_calls:
            return False
        return len(issued) < 2 * max(len(collected), PENALTY_SAFE_MIN_SEGMENTS)

    def _seed_queries(self, case: LegalCase) -> List[str]:
        ordered: List[str] = [case.case_query]
        ordered.extend(self._probe_queries)
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
        case: LegalCase,
        issued: Set[str],
        collected: Dict[str, RetrievedChunk],
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


class LawEvidenceRetriever(ABC):
    @abstractmethod
    def retrieve(self, case: LegalCase, evidence: Sequence[RetrievedChunk]) -> List[LawReference]:
        raise NotImplementedError


class Bm25LlmLawRetriever(LawEvidenceRetriever):
    _SYSTEM = (
        "Bạn là chuyên gia pháp luật Việt Nam. Hãy chọn những điều luật thực sự áp dụng cho vụ án "
        "trong danh sách ứng viên được cung cấp."
    )

    def __init__(
        self,
        model: LanguageModel,
        articles: Sequence[LawArticle],
        index: Bm25Index,
        shortlist_size: int,
        max_results: int,
    ) -> None:
        self._model = model
        self._articles = articles
        self._index = index
        self._shortlist_size = shortlist_size
        self._max_results = max_results

    def retrieve(self, case: LegalCase, evidence: Sequence[RetrievedChunk]) -> List[LawReference]:
        query = self._build_query(case, evidence)
        shortlist = [self._articles[index] for index, _ in self._index.search(query, self._shortlist_size)]
        if not shortlist:
            return []
        selected = self._judge(case, shortlist)
        return selected[: self._max_results]

    @staticmethod
    def _build_query(case: LegalCase, evidence: Sequence[RetrievedChunk]) -> str:
        parts = [case.case_query]
        parts.extend(chunk.text for chunk in evidence)
        return " ".join(parts)

    def _judge(self, case: LegalCase, shortlist: Sequence[LawArticle]) -> List[LawReference]:
        catalogue = "\n".join(
            f"[{position}] {article.law_id} | Điều {article.aid}: {article.content[:320]}"
            for position, article in enumerate(shortlist)
        )
        user_prompt = (
            f"Mô tả vụ án:\n{case.case_query}\n\n"
            f"Danh sách điều luật ứng viên:\n{catalogue}\n\n"
            "Hãy chọn các chỉ số của điều luật áp dụng trực tiếp cho vụ án, sắp theo mức độ liên quan giảm dần.\n"
            'Chỉ trả về JSON dạng: {"selected": [chỉ_số, ...]}'
        )
        payload = JsonExtractor.extract_object(self._model.generate(self._SYSTEM, user_prompt))
        indices = self._read_indices(payload, len(shortlist))
        if not indices:
            indices = list(range(min(self._max_results, len(shortlist))))
        return [LawReference(law_id=shortlist[i].law_id, aid=shortlist[i].aid) for i in indices]

    @staticmethod
    def _read_indices(payload: Optional[Dict[str, object]], upper_bound: int) -> List[int]:
        if not payload or not isinstance(payload.get("selected"), list):
            return []
        indices: List[int] = []
        for item in payload["selected"]:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= value < upper_bound and value not in indices:
                indices.append(value)
        return indices


class LawNameResolver:
    _CODE_PATTERN = re.compile(r"\d+/\d{4}/[A-Za-zĐđ\-]+")
    _OUTDATED_YEARS: Tuple[str, ...] = ("1987", "1993", "1995", "1998", "2000", "2003", "2004", "2005", "2006", "2009")

    def __init__(self, corpus_law_ids: Set[str]) -> None:
        self._corpus_law_ids = corpus_law_ids

    def resolve(self, name: str) -> Optional[str]:
        lowered = name.lower()
        code = self._match_code(name)
        if code is not None:
            return code
        return self._match_keywords(lowered)

    def _match_code(self, name: str) -> Optional[str]:
        for match in self._CODE_PATTERN.finditer(name):
            if match.group(0) in self._corpus_law_ids:
                return match.group(0)
        return None

    def _match_keywords(self, lowered: str) -> Optional[str]:
        if "blttds" in lowered:
            return "92/2015/QH13"
        if "blds" in lowered:
            return None if self._is_outdated(lowered) else "91/2015/QH13"
        if "blhs" in lowered:
            return "100/2015/QH13"
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
        if "khiếu nại" in lowered:
            return None if ("tố cáo" in lowered or self._is_outdated(lowered)) else "02/2011/QH13"
        if "tổ chức tín dụng" in lowered:
            return "47/2010/QH12"
        if "kinh doanh bất động sản" in lowered:
            return None if self._is_outdated(lowered) else "66/2014/QH13"
        if "xây dựng" in lowered:
            return "50/2014/QH13"
        if "người cao tuổi" in lowered:
            return "39/2009/QH12"
        if "nuôi con nuôi" in lowered:
            return "52/2010/QH12"
        return None

    def _is_outdated(self, lowered: str) -> bool:
        return any(year in lowered for year in self._OUTDATED_YEARS)


class ArticleNumberIndex:
    def __init__(self, articles: Sequence[LawArticle]) -> None:
        self._number_to_aid: Dict[Tuple[str, int], int] = {}
        counters: Dict[str, int] = {}
        for article in articles:
            counters[article.law_id] = counters.get(article.law_id, 0) + 1
            self._number_to_aid[(article.law_id, counters[article.law_id])] = article.aid

    def to_aid(self, law_id: str, article_no: int) -> Optional[int]:
        return self._number_to_aid.get((law_id, article_no))


class CitationLawExtractor(LawEvidenceRetriever):
    _CITATION_SPAN = re.compile(
        r"((?:(?:kho[aả]n\s+\d+\s+)?[Đđ]i[eề]u\s+\d+[\s,;và]*)+)\s*(?:c[uủ]a\s+)?"
        r"((?:B[ộo]\s+lu[ậa]t|Lu[ậa]t|Ngh[ịi]\s+quy[ếe]t|Ngh[ịi]\s+đ[ịi]nh|Ph[áa]p\s+l[ệe]nh)[^;.\n]{0,50})",
        re.IGNORECASE,
    )
    _ARTICLE_NUMBER = re.compile(r"[Đđ]i[eề]u\s+(\d+)")

    def __init__(
        self,
        resolver: LawNameResolver,
        index: ArticleNumberIndex,
        max_results: int,
        fallback: Optional[LawEvidenceRetriever] = None,
    ) -> None:
        self._resolver = resolver
        self._index = index
        self._max_results = max_results
        self._fallback = fallback

    def retrieve(self, case: LegalCase, evidence: Sequence[RetrievedChunk]) -> List[LawReference]:
        references = self._from_citations(evidence)
        if not references and self._fallback is not None:
            return self._fallback.retrieve(case, evidence)
        return references[: self._max_results]

    def _from_citations(self, evidence: Sequence[RetrievedChunk]) -> List[LawReference]:
        text = "\n".join(chunk.text for chunk in evidence)
        references: List[LawReference] = []
        seen: Set[Tuple[str, int]] = set()
        for match in self._CITATION_SPAN.finditer(text):
            law_id = self._resolver.resolve(match.group(2))
            if law_id is None:
                continue
            for raw_number in self._ARTICLE_NUMBER.findall(match.group(1)):
                aid = self._index.to_aid(law_id, int(raw_number))
                if aid is None:
                    continue
                key = (law_id, aid)
                if key not in seen:
                    seen.add(key)
                    references.append(LawReference(law_id=law_id, aid=aid))
        return references


class OutcomePredictor(ABC):
    @abstractmethod
    def predict(
        self,
        case: LegalCase,
        evidence: Sequence[RetrievedChunk],
        laws: Sequence[LawReference],
    ) -> str:
        raise NotImplementedError


class LlmOutcomePredictor(OutcomePredictor):
    _SYSTEM = (
        "Bạn là thẩm phán mô phỏng. Các đoạn bằng chứng dưới đây được trích từ chính bản án của vụ việc, "
        "bao gồm phần Nhận định và Quyết định của Tòa án. Hãy ĐỌC kỹ phần quyết định để xác định kết quả "
        "cho nguyên đơn (bên A) so với bị đơn (bên B), không suy đoán chủ quan."
    )

    def __init__(self, model: LanguageModel, articles: Sequence[LawArticle], samples: int = 1) -> None:
        self._model = model
        self._samples = max(1, samples)
        self._article_lookup = {(article.law_id, article.aid): article for article in articles}

    def predict(
        self,
        case: LegalCase,
        evidence: Sequence[RetrievedChunk],
        laws: Sequence[LawReference],
    ) -> str:
        user_prompt = self._build_prompt(case, evidence, laws)
        votes: List[str] = []
        for _ in range(self._samples):
            payload = JsonExtractor.extract_object(self._model.generate(self._SYSTEM, user_prompt))
            label = str(payload.get("prediction")).strip().upper() if payload else ""
            if label in PREDICTION_LABELS:
                votes.append(label)
        if not votes:
            return DEFAULT_PREDICTION
        return Counter(votes).most_common(1)[0][0]

    def _build_prompt(
        self,
        case: LegalCase,
        evidence: Sequence[RetrievedChunk],
        laws: Sequence[LawReference],
    ) -> str:
        evidence_text = "\n\n".join(f"- {chunk.text[:600]}" for chunk in evidence) or "(không truy xuất được bằng chứng)"
        law_text = "\n".join(self._format_law(ref) for ref in laws) or "(không xác định điều luật)"
        return (
            f"Vai trò bên A: {case.plaintiff_role}. Vai trò bên B: {case.defendant_role}.\n\n"
            f"Mô tả vụ án:\n{case.case_query}\n\n"
            f"Các đoạn trích từ bản án:\n{evidence_text}\n\n"
            f"Điều luật được viện dẫn:\n{law_text}\n\n"
            "Đọc kỹ phần Quyết định (thường đánh số 1, 2, 3...) và áp dụng quy tắc sau:\n"
            "- Nếu xuất hiện cụm 'Chấp nhận MỘT PHẦN yêu cầu', HOẶC có BẤT KỲ yêu cầu nào của nguyên đơn "
            "bị 'không chấp nhận' / 'bác' → đây là THẮNG MỘT PHẦN (PARTIAL). TUYỆT ĐỐI không chọn A_WIN "
            "dù số tiền được tuyên có lớn hay câu đầu ghi 'chấp nhận'.\n"
            "- Chỉ chọn A_WIN khi Tòa chấp nhận TOÀN BỘ yêu cầu, không bác bất kỳ phần nào.\n"
            "- Chỉ chọn B_WIN khi Tòa bác TOÀN BỘ yêu cầu của nguyên đơn.\n"
            "Với trường hợp PARTIAL, ước lượng phần nguyên đơn được chấp nhận so với tổng yêu cầu ban đầu:\n"
            "- PARTIAL_A_WIN: nguyên đơn được chấp nhận TRÊN 50%.\n"
            "- PARTIAL_B_WIN: nguyên đơn được chấp nhận KHÔNG QUÁ 50%.\n"
            "Chọn đúng một nhãn trong: A_WIN, PARTIAL_A_WIN, PARTIAL_B_WIN, B_WIN.\n"
            'Chỉ trả về JSON dạng: {"prediction": "NHÃN"}'
        )

    def _format_law(self, ref: LawReference) -> str:
        article = self._article_lookup.get((ref.law_id, ref.aid))
        content = article.content[:200] if article else ""
        return f"- {ref.law_id} | Điều {ref.aid}: {content}"


class CaseRepository:
    @staticmethod
    def load(path: str) -> List[LegalCase]:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        cases: List[LegalCase] = []
        for record in records:
            cases.append(
                LegalCase(
                    case_id=str(record.get("case_id", "")),
                    case_query=str(record.get("case_query", "")),
                    plaintiff_role=str(record.get("A_role", "Nguyên đơn")),
                    defendant_role=str(record.get("B_role", "Bị đơn")),
                )
            )
        return cases


class LawRepository:
    @staticmethod
    def load(path: str) -> List[LawArticle]:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        articles: List[LawArticle] = []
        for law in records:
            law_id = str(law.get("law_id", ""))
            for entry in law.get("content", []):
                try:
                    aid = int(entry.get("aid"))
                except (TypeError, ValueError):
                    continue
                articles.append(
                    LawArticle(law_id=law_id, aid=aid, content=str(entry.get("content_Article", "")))
                )
        return articles


class SubmissionWriter:
    @staticmethod
    def write(path: str, records: Sequence[SubmissionRecord]) -> None:
        payload = [record.to_dict() for record in records]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


class DeepSearcherPipeline:
    def __init__(
        self,
        evidence_collector: CaseEvidenceCollector,
        law_retriever: LawEvidenceRetriever,
        predictor: OutcomePredictor,
    ) -> None:
        self._evidence_collector = evidence_collector
        self._law_retriever = law_retriever
        self._predictor = predictor

    def run_case(self, case: LegalCase) -> SubmissionRecord:
        evidence = self._evidence_collector.collect(case)
        laws = self._law_retriever.retrieve(case, evidence)
        prediction = self._predictor.predict(case, evidence, laws)
        return SubmissionRecord(
            case_id=case.case_id,
            prediction=prediction,
            case_evidence=[chunk.chunk_id for chunk in evidence],
            law_evidence=laws,
        )

    def run(self, cases: Sequence[LegalCase]) -> List[SubmissionRecord]:
        records: List[SubmissionRecord] = []
        for position, case in enumerate(cases, start=1):
            print(f"[{position}/{len(cases)}] {case.case_id}", flush=True)
            records.append(self.run_case(case))
        return records


class CaseSourceFactory:
    @staticmethod
    def create(config: AppConfig) -> CaseSegmentSource:
        if not config.api_key:
            print("Không có API key cho Case Content API -> chạy offline (case_evidence rỗng).", flush=True)
            return NullCaseSegmentSource()
        return HttpCaseSegmentSource(
            api_url=config.api_url,
            api_key=config.api_key,
            request_interval=config.request_interval,
            request_timeout=config.request_timeout,
            max_retries=config.max_retries,
            rate_limit_delay=config.rate_limit_delay,
            max_rate_limit_retries=config.max_rate_limit_retries,
        )


def fingerprint_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


class ArtifactCache:
    def __init__(self, cache_dir: str, read_cache: bool, write_cache: bool) -> None:
        self._cache_dir = cache_dir
        self._read_cache = read_cache
        self._write_cache = write_cache

    def load_or_build(self, key: str, builder: Callable[[], CacheableT]) -> CacheableT:
        path = self._path(key)
        if self._read_cache:
            cached = self._load(path)
            if cached is not None:
                print(f"Dùng lại index từ cache: {path}", flush=True)
                return cached
        artifact = builder()
        if self._write_cache:
            self._save(path, artifact)
        return artifact

    def _path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self._cache_dir, f"{digest}.pkl")

    @staticmethod
    def _load(path: str) -> Optional[object]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as handle:
                return pickle.load(handle)
        except Exception:
            return None

    def _save(self, path: str, artifact: object) -> None:
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            with open(path, "wb") as handle:
                pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Đã lưu index vào cache: {path}", flush=True)
        except Exception:
            pass


class PipelineBuilder:
    @staticmethod
    def _make_model(config: AppConfig, temperature: float) -> LanguageModel:
        primary = LiteLlmLanguageModel(
            model_name=config.llm_model,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            temperature=temperature,
            max_retries=config.llm_max_retries,
            min_interval=config.llm_min_interval,
            retry_delay=config.llm_retry_delay,
        )
        if not config.use_fallback or config.fallback_model == config.llm_model:
            return primary
        fallback = LiteLlmLanguageModel(
            model_name=config.fallback_model,
            base_url=config.fallback_base_url,
            api_key=config.llm_api_key,
            temperature=temperature,
            max_retries=config.llm_max_retries,
            min_interval=0.0,
            retry_delay=config.llm_retry_delay,
        )
        return FallbackLanguageModel(primary, fallback)

    @staticmethod
    def build(config: AppConfig, articles: Sequence[LawArticle]) -> DeepSearcherPipeline:
        model = PipelineBuilder._make_model(config, config.llm_temperature)
        cache = ArtifactCache(config.cache_dir, config.read_cache, config.write_cache)
        fingerprint = fingerprint_file(config.law_path)
        texts = [article.content for article in articles]
        index = cache.load_or_build(
            f"bm25|{CACHE_VERSION}|{fingerprint}",
            lambda: Bm25Index(texts),
        )
        planner = SubQueryPlanner(model=model, queries_per_round=config.subqueries_per_round)
        collector = CaseEvidenceCollector(
            source=CaseSourceFactory.create(config),
            planner=planner,
            probe_queries=JUDGMENT_PROBE_QUERIES,
            max_calls=config.max_case_calls,
        )
        corpus_law_ids = {article.law_id for article in articles}
        bm25_fallback = Bm25LlmLawRetriever(
            model=model,
            articles=articles,
            index=index,
            shortlist_size=config.law_shortlist_size,
            max_results=config.max_law_evidence,
        )
        law_retriever = CitationLawExtractor(
            resolver=LawNameResolver(corpus_law_ids=corpus_law_ids),
            index=ArticleNumberIndex(articles),
            max_results=config.max_law_evidence,
            fallback=bm25_fallback,
        )
        outcome_model = model
        if config.outcome_samples > 1:
            outcome_model = PipelineBuilder._make_model(config, config.outcome_temperature)
        predictor = LlmOutcomePredictor(model=outcome_model, articles=articles, samples=config.outcome_samples)
        return DeepSearcherPipeline(collector, law_retriever, predictor)


def resolve_api_key(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    for name in TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_arguments() -> AppConfig:
    parser = argparse.ArgumentParser(description="ALQAC 2026 Deep Searcher submission generator.")
    parser.add_argument("--test-path", default="data/ALQAC2026_public_test.json")
    parser.add_argument("--law-path", default="data/corpus_law_pub.json")
    parser.add_argument("--output-path", default="submission.json")
    parser.add_argument("--api-url", default="https://alqac-api.ngrok.pro/retrieve")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--request-interval", type=float, default=6.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rate-limit-delay", type=float, default=7.0)
    parser.add_argument("--max-rate-limit-retries", type=int, default=10)
    parser.add_argument("--max-case-calls", type=int, default=20)
    parser.add_argument("--subqueries-per-round", type=int, default=4)
    parser.add_argument("--law-shortlist-size", type=int, default=20)
    parser.add_argument("--max-law-evidence", type=int, default=12)
    parser.add_argument("--llm-model", default="azure/gpt-4o")
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-retries", type=int, default=3)
    parser.add_argument("--llm-min-interval", type=float, default=10.0)
    parser.add_argument("--llm-retry-delay", type=float, default=5.0)
    parser.add_argument("--fallback-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--fallback-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--outcome-samples", type=int, default=1)
    parser.add_argument("--outcome-temperature", type=float, default=0.6)
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parsed = parser.parse_args()
    api_key = None if parsed.offline else resolve_api_key(parsed.api_key)
    return AppConfig(
        test_path=parsed.test_path,
        law_path=parsed.law_path,
        output_path=parsed.output_path,
        api_url=parsed.api_url,
        api_key=api_key,
        request_interval=parsed.request_interval,
        request_timeout=parsed.request_timeout,
        max_retries=parsed.max_retries,
        rate_limit_delay=parsed.rate_limit_delay,
        max_rate_limit_retries=parsed.max_rate_limit_retries,
        max_case_calls=parsed.max_case_calls,
        subqueries_per_round=parsed.subqueries_per_round,
        law_shortlist_size=parsed.law_shortlist_size,
        max_law_evidence=parsed.max_law_evidence,
        llm_model=parsed.llm_model,
        llm_base_url=parsed.llm_base_url,
        llm_api_key=parsed.llm_api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        llm_temperature=parsed.llm_temperature,
        llm_max_retries=parsed.llm_max_retries,
        llm_min_interval=parsed.llm_min_interval,
        llm_retry_delay=parsed.llm_retry_delay,
        fallback_model=parsed.fallback_model,
        fallback_base_url=parsed.fallback_base_url,
        use_fallback=not parsed.no_fallback,
        outcome_samples=parsed.outcome_samples,
        outcome_temperature=parsed.outcome_temperature,
        cache_dir=parsed.cache_dir,
        read_cache=not parsed.no_cache and not parsed.rebuild_cache,
        write_cache=not parsed.no_cache,
    )


def main() -> None:
    config = parse_arguments()
    cases = CaseRepository.load(config.test_path)
    articles = LawRepository.load(config.law_path)
    print(f"Đã nạp {len(cases)} vụ án và {len(articles)} điều luật.", flush=True)
    pipeline = PipelineBuilder.build(config, articles)
    records = pipeline.run(cases)
    SubmissionWriter.write(config.output_path, records)
    print(f"Đã ghi {len(records)} kết quả vào {config.output_path}.", flush=True)


if __name__ == "__main__":
    main()
