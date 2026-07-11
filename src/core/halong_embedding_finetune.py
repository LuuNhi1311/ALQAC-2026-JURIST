from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/media/caotulab/303A225B3A221DFA/hf_cache")
os.environ.setdefault("WANDB_PROJECT", "alqac-halong-finetune")

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import wandb
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
from sentence_transformers.util import cos_sim, mine_hard_negatives

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORPUS_PATH = os.environ.get("ALQAC_CORPUS", str(DATA_DIR / "corpus_law_pub.json"))
TEST_PATH = os.environ.get("ALQAC_TEST", str(DATA_DIR / "ALQAC2026_public_test.json"))
OUTPUT_DIR = os.environ.get("ALQAC_OUTPUT_DIR", "/media/caotulab/303A225B3A221DFA/alqac_halong_embedding")
MODEL_NAME = "hiieu/halong_embedding"
HUB_MODEL_ID = "leonpham1208/alqac_halong_embedding"
HUB_LICENSE = os.environ.get("ALQAC_HUB_LICENSE", "apache-2.0")

EVAL_HELDOUT_CASES = 10
ICT_SENTENCES_PER_ARTICLE = 1
MIN_SENTENCE_LEN = 25
MATRYOSHKA_DIMENSIONS = [768, 512, 256, 128, 64]
MAX_SEQ_LENGTH = 512
NUM_TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
K_VALUES = [1, 3, 5, 10]

_CODE_RE = re.compile(r"\d+/\d{4}/[A-Za-zĐđ\-]+")
_ART_RE = re.compile(r"[Đđ]iều\s+(\d+)")
_OUTDATED = ("1987", "1993", "1995", "1998", "2000", "2003", "2004", "2005", "2006", "2009")


@dataclass
class DataBundle:
    corpus: Dict[str, str]
    queries: Dict[str, str]
    relevant_docs: Dict[str, Set[str]]
    alqac_records: List[Tuple[str, str, str]]
    ict_records: List[Tuple[str, str, str]]


def resolve_law_id(name: str, corpus_law_ids: Set[str]) -> Optional[str]:
    for match in _CODE_RE.finditer(name):
        if match.group(0) in corpus_law_ids:
            return match.group(0)
    low = name.lower()
    outdated = any(year in low for year in _OUTDATED)
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


def cited_cids(
    related_text: str,
    corpus: Dict[str, str],
    number_to_aid: Dict[Tuple[str, int], int],
    corpus_law_ids: Set[str],
) -> Set[str]:
    cids: Set[str] = set()
    for line in str(related_text or "").splitlines():
        if "|" not in line:
            continue
        name, remainder = line.split("|", 1)
        law_id = resolve_law_id(name.strip(), corpus_law_ids)
        if law_id is None:
            continue
        for raw_number in _ART_RE.findall(remainder):
            aid = number_to_aid.get((law_id, int(raw_number)))
            cid = f"{law_id}::{aid}" if aid is not None else None
            if cid in corpus:
                cids.add(cid)
    return cids


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.;\n])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) >= MIN_SENTENCE_LEN]


def load_corpus(corpus_path: str) -> Tuple[Dict[str, str], Dict[Tuple[str, int], int]]:
    with open(corpus_path, "r", encoding="utf-8") as handle:
        documents = json.load(handle)
    corpus: Dict[str, str] = {}
    number_to_aid: Dict[Tuple[str, int], int] = {}
    for law in documents:
        law_id = str(law.get("law_id", "")).strip()
        article_no = 0
        for article in law.get("content", []):
            article_no += 1
            number_to_aid[(law_id, article_no)] = article["aid"]
            text = str(article.get("content_Article", "")).strip()
            if text:
                corpus[f"{law_id}::{article['aid']}"] = text
    return corpus, number_to_aid


def prepare_data() -> DataBundle:
    random.seed(42)
    corpus, number_to_aid = load_corpus(CORPUS_PATH)
    corpus_law_ids = {cid.split("::")[0] for cid in corpus}

    labelled_cases: List[Tuple[str, str, Set[str]]] = []
    if os.path.exists(TEST_PATH):
        with open(TEST_PATH, "r", encoding="utf-8") as handle:
            cases = json.load(handle)
        for case in cases:
            query = str(case.get("case_query", "")).strip()
            cids = cited_cids(case.get("related_law_provisions", ""), corpus, number_to_aid, corpus_law_ids)
            if query and cids:
                labelled_cases.append((str(case.get("case_id", "")), query, cids))

    random.shuffle(labelled_cases)
    eval_cases = labelled_cases[:EVAL_HELDOUT_CASES] if EVAL_HELDOUT_CASES > 0 else []
    train_cases = labelled_cases[EVAL_HELDOUT_CASES:] if EVAL_HELDOUT_CASES > 0 else labelled_cases

    alqac_records = [(query, corpus[cid], cid) for (_id, query, cids) in train_cases for cid in cids]

    ict_records: List[Tuple[str, str, str]] = []
    for cid, text in corpus.items():
        candidate_sentences = sentences(text)
        if not candidate_sentences:
            continue
        for sentence in random.sample(candidate_sentences, min(ICT_SENTENCES_PER_ARTICLE, len(candidate_sentences))):
            ict_records.append((sentence, text, cid))

    if eval_cases:
        queries = {cid_key: query for (cid_key, query, _cids) in eval_cases}
        relevant_docs = {cid_key: set(cids) for (cid_key, _q, cids) in eval_cases}
    else:
        random.shuffle(ict_records)
        hold = ict_records[:1000]
        queries = {str(index): anchor for index, (anchor, _p, _c) in enumerate(hold)}
        relevant_docs = {str(index): {cid} for index, (_a, _p, cid) in enumerate(hold)}

    print(f"corpus: {len(corpus)} articles | labelled cases: {len(labelled_cases)}")
    print(f"train cases: {len(train_cases)} -> supervised {len(alqac_records)} | ICT {len(ict_records)}")
    print(f"eval queries: {len(queries)}")
    return DataBundle(corpus, queries, relevant_docs, alqac_records, ict_records)


def load_model() -> SentenceTransformer:
    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def build_evaluator(data: DataBundle) -> SequentialEvaluator:
    evaluators = []
    for dim in MATRYOSHKA_DIMENSIONS:
        evaluators.append(
            InformationRetrievalEvaluator(
                queries=data.queries,
                corpus=data.corpus,
                relevant_docs=data.relevant_docs,
                name=f"dim_{dim}",
                truncate_dim=dim,
                score_functions={"cosine": cos_sim},
            )
        )
    return SequentialEvaluator(evaluators)


def build_train_dataset(model: SentenceTransformer, data: DataBundle) -> Dict[str, Dataset]:
    alqac_dataset = Dataset.from_dict(
        {
            "anchor": [anchor for (anchor, _pos, _cid) in data.alqac_records],
            "positive": [positive for (_a, positive, _cid) in data.alqac_records],
        }
    )
    corpus_documents = list(dict.fromkeys(data.corpus.values()))
    alqac_triplets = mine_hard_negatives(
        alqac_dataset,
        model,
        corpus=corpus_documents,
        num_negatives=5,
        range_min=0,
        range_max=100,
        sampling_strategy="top",
        batch_size=64,
        output_format="triplet",
    )
    ict_dataset = Dataset.from_dict(
        {
            "anchor": [anchor for (anchor, _pos, _cid) in data.ict_records],
            "positive": [positive for (_a, positive, _cid) in data.ict_records],
        }
    )
    return {"alqac": alqac_triplets, "ict": ict_dataset}


def build_loss(model: SentenceTransformer) -> MatryoshkaLoss:
    inner_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
    return MatryoshkaLoss(model, inner_loss, matryoshka_dims=MATRYOSHKA_DIMENSIONS)


def build_training_args() -> SentenceTransformerTrainingArguments:
    return SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.ROUND_ROBIN,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_768_cosine_ndcg@10",
        report_to="wandb",
        run_name="nina-alqac-embedding",
    )


def evaluate_retrieval(model: SentenceTransformer, data: DataBundle) -> None:
    corpus_ids = list(data.corpus.keys())
    corpus_texts = [data.corpus[cid] for cid in corpus_ids]
    document_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_ids = list(data.queries.keys())
    query_texts = [data.queries[qid] for qid in query_ids]
    query_embeddings = model.encode(
        query_texts,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarity = query_embeddings @ document_embeddings.T
    ranking = torch.topk(similarity, k=max(K_VALUES), dim=1).indices.cpu().numpy()

    print(f"{'K':>3} | {'Accuracy@K':>10} | {'Precision@K':>11} | {'Recall@K':>9} | {'F1@K':>7}")
    for k in K_VALUES:
        precision_at_k, recall_at_k, accuracy_at_k, f1_at_k = [], [], [], []
        for row, qid in enumerate(query_ids):
            relevant = data.relevant_docs[qid]
            retrieved = [corpus_ids[index] for index in ranking[row, :k]]
            hits = sum(1 for cid in retrieved if cid in relevant)
            precision = hits / k
            recall = hits / len(relevant) if relevant else 0.0
            accuracy = 1.0 if hits > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            precision_at_k.append(precision)
            recall_at_k.append(recall)
            accuracy_at_k.append(accuracy)
            f1_at_k.append(f1)
        print(
            f"{k:>3} | {float(np.mean(accuracy_at_k)):>10.4f} | {float(np.mean(precision_at_k)):>11.4f} | "
            f"{float(np.mean(recall_at_k)):>9.4f} | {float(np.mean(f1_at_k)):>7.4f}"
        )


def print_metrics(results: Dict[str, float]) -> None:
    for key, value in results.items():
        print(key, value)


def load_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("HF_TOKEN"):
            continue
        _, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            return value
    return None


def format_metrics_table(metrics: Dict[str, float]) -> str:
    patterns = ("ndcg@", "mrr@", "map@", "accuracy@", "precision@", "recall@")
    selected = {
        key: value
        for key, value in metrics.items()
        if any(pattern in key for pattern in patterns)
    }

    if not selected:
        return "No evaluation results were recorded."

    rows = ["| Metric | Value |", "| --- | --- |"]
    for key in sorted(selected):
        try:
            rows.append(f"| {key} | {float(selected[key]):.4f} |")
        except (TypeError, ValueError):
            continue
    return "\n".join(rows)


def build_model_card(metrics: Dict[str, float]) -> str:
    metrics_block = format_metrics_table(metrics)
    dims = ", ".join(str(dim) for dim in MATRYOSHKA_DIMENSIONS)

    frontmatter = "\n".join(
        [
            "---",
            "language:",
            "- vi",
            f"license: {HUB_LICENSE}",
            "library_name: sentence-transformers",
            "pipeline_tag: sentence-similarity",
            "tags:",
            "- sentence-transformers",
            "- sentence-similarity",
            "- feature-extraction",
            "- matryoshka",
            "- legal",
            "- vietnamese",
            f"base_model: {MODEL_NAME}",
            "---",
        ]
    )

    body = f"""
# ALQAC Halong Legal Embedding

A **Vietnamese legal retrieval embedding model**, fine-tuned from
`{MODEL_NAME}`. It retrieves the **relevant law articles** for a legal
question or case description (dense retrieval).

## Training

- **Base model**: `{MODEL_NAME}` (max_seq_length = {MAX_SEQ_LENGTH}).
- **Data**:
  - ALQAC supervised pairs: `case_query` → cited law articles
    (`related_law_provisions`).
  - ICT (Inverse Cloze Task) pairs synthesized from the law corpus for
    additional self-supervised signal.
- **Loss**: `MatryoshkaLoss(CachedMultipleNegativesRankingLoss)` over the
  Matryoshka dimensions `[{dims}]`, so the embedding can be truncated to fewer
  dimensions while staying effective.
- **Hard negatives**: mined with `mine_hard_negatives`.
- Best checkpoint selected by `ndcg@10` (dimension 768).

## Results (retrieval on the held-out ALQAC split)

{metrics_block}

## Usage

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("{HUB_MODEL_ID}")

query = "How is compensation for damage caused by animals regulated?"
documents = [
    "Article 603. Compensation for damage caused by animals ...",
    "Article 584. Basis for arising liability to compensate for damage ...",
]

query_emb = model.encode(query, normalize_embeddings=True)
doc_emb = model.encode(documents, normalize_embeddings=True)

scores = cos_sim(query_emb, doc_emb)[0]
for doc, score in sorted(zip(documents, scores), key=lambda pair: -pair[1]):
    print(round(float(score), 4), doc[:60])
```

Use fewer dimensions with Matryoshka (faster / lighter):

```python
model = SentenceTransformer("{HUB_MODEL_ID}", truncate_dim=256)
emb = model.encode("a legal question", normalize_embeddings=True)
print(emb.shape)  # (256,)
```

## Limitations

- Trained on the small-scale ALQAC 2026 dataset, tuned for the Vietnamese
  legal domain.
- Retrieval quality depends on the accompanying law-article corpus.
"""

    return frontmatter + "\n" + body.strip() + "\n"


def push_to_hub(
    model: SentenceTransformer,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    token = load_hf_token()
    if not token:
        print("Không tìm thấy HF_TOKEN trong .env, bỏ qua push lên hub.")
        return

    from huggingface_hub import login, upload_file

    login(token=token)
    model.push_to_hub(
        HUB_MODEL_ID,
        private=True,
        exist_ok=True,
        train_datasets=["ALQAC2026_public_test", "corpus_law_pub"],
    )

    readme_path = Path(OUTPUT_DIR) / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(build_model_card(metrics or {}), encoding="utf-8")

    upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=HUB_MODEL_ID,
        repo_type="model",
    )

    print(f"Đã push model + model card lên hub: {HUB_MODEL_ID}")


def main() -> None:
    wandb.login()
    data = prepare_data()
    model = load_model()
    evaluator = build_evaluator(data)
    print_metrics(evaluator(model))
    train_dataset = build_train_dataset(model, data)
    loss = build_loss(model)
    args = build_training_args()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    trainer.save_model()
    print(f"Đã lưu model tuned tại local: {args.output_dir}")
    fine_tuned_model = SentenceTransformer(
        args.output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    final_metrics = evaluator(fine_tuned_model)
    print_metrics(final_metrics)
    evaluate_retrieval(fine_tuned_model, data)
    push_to_hub(fine_tuned_model, final_metrics)


if __name__ == "__main__":
    main()
