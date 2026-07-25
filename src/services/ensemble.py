from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

PREDICTION_LABELS: Tuple[str, ...] = ("A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN")
DEFAULT_PREDICTION: str = "PARTIAL_A_WIN"


@dataclass(frozen=True)
class LawReference:
    law_id: str
    aid: int


@dataclass(frozen=True)
class PredictedCase:
    case_id: str
    prediction: str
    case_evidence: List[str]
    law_evidence: List[LawReference]


@dataclass
class MergedCase:
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


class SubmissionLoader:
    @staticmethod
    def load(path: str) -> Dict[str, PredictedCase]:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        cases: Dict[str, PredictedCase] = {}
        for record in records:
            case_id = str(record.get("case_id", ""))
            cases[case_id] = PredictedCase(
                case_id=case_id,
                prediction=str(record.get("prediction", "")).strip().upper(),
                case_evidence=[str(item) for item in record.get("case_evidence", []) or []],
                law_evidence=SubmissionLoader._law(record.get("law_evidence", [])),
            )
        return cases

    @staticmethod
    def _law(raw: object) -> List[LawReference]:
        references: List[LawReference] = []
        if not isinstance(raw, list):
            return references
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                aid = int(item.get("aid"))
            except (TypeError, ValueError):
                continue
            references.append(LawReference(str(item.get("law_id", "")), aid))
        return references


class OutcomeVoter:
    def __init__(self, primary_position: int) -> None:
        self._primary_position = primary_position

    def vote(self, predictions: Sequence[str]) -> str:
        """Majority vote over pipeline predictions; ties broken toward the primary pipeline's label."""
        valid = [prediction for prediction in predictions if prediction in PREDICTION_LABELS]
        if not valid:
            return DEFAULT_PREDICTION
        counts = Counter(valid)
        top_count = counts.most_common(1)[0][1]
        leaders = [label for label, count in counts.items() if count == top_count]
        if len(leaders) == 1:
            return leaders[0]
        primary = predictions[self._primary_position] if self._primary_position < len(predictions) else ""
        if primary in leaders:
            return primary
        return leaders[0]


class EvidenceUnion:
    @staticmethod
    def case_evidence(sources: Sequence[List[str]]) -> List[str]:
        merged: List[str] = []
        seen: set = set()
        for evidence in sources:
            for chunk_id in evidence:
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    merged.append(chunk_id)
        return merged

    @staticmethod
    def law_evidence(sources: Sequence[List[LawReference]]) -> List[LawReference]:
        merged: List[LawReference] = []
        seen: set = set()
        for references in sources:
            for reference in references:
                key = (reference.law_id, reference.aid)
                if key not in seen:
                    seen.add(key)
                    merged.append(reference)
        return merged


class EnsembleMerger:
    def __init__(self, voter: OutcomeVoter) -> None:
        self._voter = voter

    def merge(self, submissions: Sequence[Dict[str, PredictedCase]]) -> List[MergedCase]:
        """Merge submissions per case: vote the verdict, union the case/law evidence across pipelines."""
        case_ids = self._ordered_case_ids(submissions)
        merged: List[MergedCase] = []
        for case_id in case_ids:
            present = [submission[case_id] for submission in submissions if case_id in submission]
            merged.append(
                MergedCase(
                    case_id=case_id,
                    prediction=self._voter.vote([case.prediction for case in present]),
                    case_evidence=EvidenceUnion.case_evidence([case.case_evidence for case in present]),
                    law_evidence=EvidenceUnion.law_evidence([case.law_evidence for case in present]),
                )
            )
        return merged

    @staticmethod
    def _ordered_case_ids(submissions: Sequence[Dict[str, PredictedCase]]) -> List[str]:
        ordered: List[str] = []
        seen: set = set()
        for submission in submissions:
            for case_id in submission:
                if case_id not in seen:
                    seen.add(case_id)
                    ordered.append(case_id)
        return ordered


class SubmissionWriter:
    @staticmethod
    def write(path: str, cases: Sequence[MergedCase]) -> None:
        payload = [case.to_dict() for case in cases]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALQAC 2026 ensemble merger for submission.json files.")
    parser.add_argument("submissions", nargs="+")
    parser.add_argument("--output-path", default="submission_ensemble.json")
    parser.add_argument("--primary", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    submissions = [SubmissionLoader.load(path) for path in arguments.submissions]
    voter = OutcomeVoter(primary_position=arguments.primary)
    merger = EnsembleMerger(voter=voter)
    merged = merger.merge(submissions)
    SubmissionWriter.write(arguments.output_path, merged)
    print(
        f"Đã gộp {len(submissions)} submission thành {len(merged)} vụ -> {arguments.output_path} "
        f"(tie-break: {arguments.submissions[arguments.primary]}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
