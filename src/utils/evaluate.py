from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

PREDICTION_LABELS: Tuple[str, ...] = ("A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN")
WEIGHT_OUTCOME: float = 0.70
WEIGHT_CASE_RECALL: float = 0.20
WEIGHT_LAW_F1: float = 0.10


@dataclass(frozen=True)
class LawKey:
    law_id: str
    aid: int


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    verdict_label: str
    law_keys: Set[LawKey]
    unresolved_provisions: int


@dataclass(frozen=True)
class PredictedCase:
    case_id: str
    prediction: str
    law_keys: Set[LawKey]
    case_evidence_count: int


@dataclass
class OutcomeReport:
    total: int = 0
    correct: int = 0
    per_label_gold: Dict[str, int] = field(default_factory=dict)
    per_label_correct: Dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class LawReport:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    macro_f1_sum: float = 0.0
    scored_cases: int = 0
    unresolved_provisions: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def micro_f1(self) -> float:
        return self._f1(self.precision, self.recall)

    @property
    def macro_f1(self) -> float:
        return self.macro_f1_sum / self.scored_cases if self.scored_cases else 0.0

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


@dataclass
class SubmissionReport:
    name: str
    outcome: OutcomeReport
    law: LawReport
    average_case_evidence: float
    missing_cases: int
    case_recall: float = 0.0

    @property
    def final_score(self) -> float:
        return (
            WEIGHT_OUTCOME * self.outcome.accuracy
            + WEIGHT_CASE_RECALL * self.case_recall
            + WEIGHT_LAW_F1 * self.law.micro_f1
        )


class LawArticleIndex:
    def __init__(self, path: str) -> None:
        self._number_to_aid: Dict[Tuple[str, int], int] = {}
        self._valid_keys: Set[LawKey] = set()
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        for law in records:
            law_id = str(law.get("law_id", ""))
            article_no = 0
            for entry in law.get("content", []):
                try:
                    aid = int(entry.get("aid"))
                except (TypeError, ValueError):
                    continue
                article_no += 1
                self._number_to_aid[(law_id, article_no)] = aid
                self._valid_keys.add(LawKey(law_id, aid))

    def resolve_number(self, law_id: str, article_no: int) -> Optional[LawKey]:
        aid = self._number_to_aid.get((law_id, article_no))
        return LawKey(law_id, aid) if aid is not None else None

    def is_valid(self, key: LawKey) -> bool:
        return key in self._valid_keys

    def law_ids(self) -> Set[str]:
        return {key.law_id for key in self._valid_keys}


class GoldLawResolver:
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


class GoldStore:
    _ARTICLE_PATTERN = re.compile(r"[Đđ]iều\s+(\d+)")

    def __init__(self, index: LawArticleIndex, resolver: GoldLawResolver) -> None:
        self._index = index
        self._resolver = resolver

    def load(self, path: str) -> Dict[str, GoldCase]:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        gold: Dict[str, GoldCase] = {}
        for record in records:
            case_id = str(record.get("case_id", ""))
            keys, unresolved = self._parse_provisions(str(record.get("related_law_provisions", "")))
            gold[case_id] = GoldCase(
                case_id=case_id,
                verdict_label=str(record.get("verdict_label", "")).strip().upper(),
                law_keys=keys,
                unresolved_provisions=unresolved,
            )
        return gold

    def _parse_provisions(self, block: str) -> Tuple[Set[LawKey], int]:
        """Parse gold "law_name | Điều N" lines into (law_id, aid) keys; count provisions outside the corpus as unresolved (unscored)."""
        keys: Set[LawKey] = set()
        unresolved = 0
        for line in block.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, remainder = line.split("|", 1)
            law_id = self._resolver.resolve(name.strip())
            numbers = [int(value) for value in self._ARTICLE_PATTERN.findall(remainder)]
            if not numbers:
                continue
            if law_id is None:
                unresolved += len(numbers)
                continue
            for number in numbers:
                key = self._index.resolve_number(law_id, number)
                if key is not None:
                    keys.add(key)
                else:
                    unresolved += 1
        return keys, unresolved


class SubmissionStore:
    def load(self, path: str) -> Dict[str, PredictedCase]:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        predictions: Dict[str, PredictedCase] = {}
        for record in records:
            case_id = str(record.get("case_id", ""))
            predictions[case_id] = PredictedCase(
                case_id=case_id,
                prediction=str(record.get("prediction", "")).strip().upper(),
                law_keys=self._parse_law_evidence(record.get("law_evidence", [])),
                case_evidence_count=len(record.get("case_evidence", []) or []),
            )
        return predictions

    @staticmethod
    def _parse_law_evidence(raw: object) -> Set[LawKey]:
        keys: Set[LawKey] = set()
        if not isinstance(raw, list):
            return keys
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                aid = int(item.get("aid"))
            except (TypeError, ValueError):
                continue
            keys.add(LawKey(str(item.get("law_id", "")), aid))
        return keys


class Evaluator:
    def __init__(self, gold: Dict[str, GoldCase]) -> None:
        self._gold = gold

    def evaluate(
        self,
        name: str,
        predictions: Dict[str, PredictedCase],
        case_recall: float = 0.0,
    ) -> SubmissionReport:
        outcome = OutcomeReport()
        law = LawReport()
        evidence_total = 0
        missing = 0
        for case_id, gold_case in self._gold.items():
            predicted = predictions.get(case_id)
            if predicted is None:
                missing += 1
                law.false_negative += len(gold_case.law_keys)
                law.unresolved_provisions += gold_case.unresolved_provisions
                self._accumulate_outcome(outcome, gold_case, None)
                continue
            evidence_total += predicted.case_evidence_count
            self._accumulate_outcome(outcome, gold_case, predicted)
            self._accumulate_law(law, gold_case, predicted)
        average_evidence = evidence_total / len(self._gold) if self._gold else 0.0
        return SubmissionReport(name, outcome, law, average_evidence, missing, case_recall)

    @staticmethod
    def _accumulate_outcome(report: OutcomeReport, gold_case: GoldCase, predicted: Optional[PredictedCase]) -> None:
        report.total += 1
        report.per_label_gold[gold_case.verdict_label] = report.per_label_gold.get(gold_case.verdict_label, 0) + 1
        if predicted is not None and predicted.prediction == gold_case.verdict_label:
            report.correct += 1
            report.per_label_correct[gold_case.verdict_label] = (
                report.per_label_correct.get(gold_case.verdict_label, 0) + 1
            )

    @staticmethod
    def _accumulate_law(report: LawReport, gold_case: GoldCase, predicted: PredictedCase) -> None:
        """Tally TP/FP/FN of predicted vs. gold law keys (feeds micro-F1) and add this case's macro-F1."""
        gold_keys = gold_case.law_keys
        predicted_keys = predicted.law_keys
        true_positive = len(gold_keys & predicted_keys)
        false_positive = len(predicted_keys - gold_keys)
        false_negative = len(gold_keys - predicted_keys)
        report.true_positive += true_positive
        report.false_positive += false_positive
        report.false_negative += false_negative
        report.unresolved_provisions += gold_case.unresolved_provisions
        report.scored_cases += 1
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        report.macro_f1_sum += LawReport._f1(precision, recall)


class ReportPrinter:
    @staticmethod
    def print_reports(reports: Sequence[SubmissionReport]) -> None:
        for report in reports:
            ReportPrinter._print_single(report)
        if len(reports) > 1:
            ReportPrinter._print_comparison(reports)

    @staticmethod
    def _print_single(report: SubmissionReport) -> None:
        print("=" * 68)
        print(f"Submission: {report.name}")
        print("-" * 68)
        outcome = report.outcome
        print(f"Outcome accuracy : {outcome.accuracy:.4f} ({outcome.correct}/{outcome.total})")
        for label in PREDICTION_LABELS:
            gold_count = outcome.per_label_gold.get(label, 0)
            correct = outcome.per_label_correct.get(label, 0)
            recall = correct / gold_count if gold_count else 0.0
            print(f"  {label:<15} recall {recall:.3f} ({correct}/{gold_count})")
        law = report.law
        print(
            f"Law evidence     : micro-F1 {law.micro_f1:.4f} | P {law.precision:.4f} "
            f"| R {law.recall:.4f} | macro-F1 {law.macro_f1:.4f}"
        )
        print(f"  TP/FP/FN       : {law.true_positive}/{law.false_positive}/{law.false_negative}")
        print(f"  Gold provisions ngoài corpus (không chấm): {law.unresolved_provisions}")
        print(f"Case evidence    : trung bình {report.average_case_evidence:.2f} chunk/vụ")
        print(
            f"                   penalized recall = {report.case_recall:.4f} "
            "(giả định, không đo được offline)"
        )
        print(
            f"FinalScore       : {report.final_score:.4f} "
            f"= {WEIGHT_OUTCOME:.2f}*{outcome.accuracy:.4f} "
            f"+ {WEIGHT_CASE_RECALL:.2f}*{report.case_recall:.4f} "
            f"+ {WEIGHT_LAW_F1:.2f}*{law.micro_f1:.4f}"
        )
        if report.missing_cases:
            print(f"Cảnh báo         : thiếu dự đoán cho {report.missing_cases} vụ án")
        print()

    @staticmethod
    def _print_comparison(reports: Sequence[SubmissionReport]) -> None:
        print("=" * 68)
        print("So sánh")
        print("-" * 68)
        header = f"{'Metric':<20}" + "".join(f"{report.name[:16]:>18}" for report in reports)
        print(header)
        rows = [
            ("Outcome accuracy", [f"{report.outcome.accuracy:.4f}" for report in reports]),
            ("Law micro-F1", [f"{report.law.micro_f1:.4f}" for report in reports]),
            ("Law macro-F1", [f"{report.law.macro_f1:.4f}" for report in reports]),
            ("Case recall", [f"{report.case_recall:.4f}" for report in reports]),
            ("FinalScore", [f"{report.final_score:.4f}" for report in reports]),
        ]
        for label, values in rows:
            print(f"{label:<20}" + "".join(f"{value:>18}" for value in values))
        print()


def api_efficiency_factor(num_calls: int, num_segments: int) -> float:
    """Case-recall penalty: full credit up to 2x calls-per-segment, linearly decaying to 0 at 5x."""
    if num_segments <= 0:
        return 0.0
    full_credit_calls = 2 * num_segments
    zero_credit_calls = 5 * num_segments
    if num_calls <= full_credit_calls:
        return 1.0
    if num_calls >= zero_credit_calls:
        return 0.0
    return (zero_credit_calls - num_calls) / (zero_credit_calls - full_credit_calls)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALQAC 2026 local scorer for submission.json files.")
    parser.add_argument("submissions", nargs="+")
    parser.add_argument("--test-path", default="data/ALQAC2026_public_test.json")
    parser.add_argument("--law-path", default="data/corpus_law_pub.json")
    parser.add_argument("--case-recall", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    index = LawArticleIndex(arguments.law_path)
    resolver = GoldLawResolver(corpus_law_ids=index.law_ids())
    gold = GoldStore(index=index, resolver=resolver).load(arguments.test_path)
    evaluator = Evaluator(gold)
    submission_store = SubmissionStore()
    reports: List[SubmissionReport] = []
    for path in arguments.submissions:
        predictions = submission_store.load(path)
        reports.append(
            evaluator.evaluate(name=path, predictions=predictions, case_recall=arguments.case_recall)
        )
    ReportPrinter.print_reports(reports)


if __name__ == "__main__":
    main()
