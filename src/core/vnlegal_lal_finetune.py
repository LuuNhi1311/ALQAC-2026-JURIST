from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/media/caotulab/303A225B3A221DFA/hf_cache")
os.environ.setdefault("WANDB_PROJECT", "alqac-vnlegal-lal")

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORPUS_PATH = os.environ.get("ALQAC_CORPUS", str(DATA_DIR / "corpus_law_pub.json"))
TEST_PATH = os.environ.get("ALQAC_TEST", str(DATA_DIR / "ALQAC2026_public_test.json"))
OUTPUT_DIR = os.environ.get("ALQAC_OUTPUT_DIR", "/media/caotulab/303A225B3A221DFA/alqac_vnlegal_lal_embedding")
MERGED_DIR = os.environ.get("ALQAC_MERGED_DIR", "/media/caotulab/303A225B3A221DFA/alqac_vnlegal_lal_embedding_merged")
MODEL_NAME = "darklethelong/vnlegal-lal"
HUB_MODEL_ID = "leonpham1208/alqac_vnlegal_lal"
HUB_LICENSE = os.environ.get("ALQAC_HUB_LICENSE", "apache-2.0")
QUERY_PROMPT = "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: "

EVAL_HELDOUT_CASES = 10
MAX_SEQ_LENGTH = 2048
K_VALUES = [1, 3, 5, 10]
ANCE_ROUNDS = 3
NEGATIVES_PER_QUERY = 8
DENSE_POOL = 100
BM25_POOL = 100
FALSE_NEGATIVE_MARGIN = 0.05
NUM_TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 32

_CODE_RE = re.compile(r"\d+/\d{4}/[A-Za-zĐđ\-]+")
_ART_RE = re.compile(r"[Đđ]iều\s+(\d+)")
_OUTDATED = ("1987", "1993", "1995", "1998", "2000", "2003", "2004", "2005", "2006", "2009")


@dataclass
class DataBundle:
    corpus: Dict[str, str]
    queries: Dict[str, str]
    relevant_docs: Dict[str, Set[str]]
    train_examples: List[Tuple[str, Set[str]]]
    corpus_ids: List[str]
    corpus_texts: List[str]
    cid_to_index: Dict[str, int]


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

    train_examples = [(query, cids) for (_id, query, cids) in train_cases]
    queries = {cid_key: query for (cid_key, query, _cids) in eval_cases}
    relevant_docs = {cid_key: set(cids) for (cid_key, _q, cids) in eval_cases}

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    cid_to_index = {cid: index for index, cid in enumerate(corpus_ids)}

    print(f"corpus: {len(corpus)} articles | labelled cases: {len(labelled_cases)}")
    print(f"train cases: {len(train_cases)} | eval cases: {len(eval_cases)}")
    return DataBundle(corpus, queries, relevant_docs, train_examples, corpus_ids, corpus_texts, cid_to_index)


def load_model() -> SentenceTransformer:
    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def build_evaluator(data: DataBundle) -> InformationRetrievalEvaluator:
    return InformationRetrievalEvaluator(
        queries=data.queries,
        corpus=data.corpus,
        relevant_docs=data.relevant_docs,
        name="alqac",
        query_prompt=QUERY_PROMPT,
        accuracy_at_k=K_VALUES,
        precision_recall_at_k=K_VALUES,
        ndcg_at_k=[10],
        mrr_at_k=[10],
        map_at_k=[10],
        show_progress_bar=True,
    )


def build_bm25(data: DataBundle) -> BM25Okapi:
    bm25 = BM25Okapi([tokenize(text) for text in data.corpus_texts])
    print(f"BM25 index built over {len(data.corpus_texts)} documents")
    return bm25


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def attach_lora(model: SentenceTransformer) -> None:
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model[0].model = get_peft_model(model[0].model, lora_config)
    model[0].model.print_trainable_parameters()


def encode_corpus(model: SentenceTransformer, corpus_texts: List[str]) -> torch.Tensor:
    return model.encode(
        corpus_texts,
        batch_size=16,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def mine_triplets(model: SentenceTransformer, data: DataBundle, bm25: BM25Okapi, document_embeddings: torch.Tensor) -> Dataset:
    query_texts = [query for query, _cids in data.train_examples]
    query_embeddings = model.encode(
        query_texts,
        prompt=QUERY_PROMPT,
        batch_size=16,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dense_scores = query_embeddings @ document_embeddings.T
    anchors, positives, negatives = [], [], []
    for row, (query, relevant_cids) in enumerate(data.train_examples):
        relevant_indices = {data.cid_to_index[cid] for cid in relevant_cids}
        best_positive_score = max(dense_scores[row, index].item() for index in relevant_indices)
        dense_candidates = torch.topk(dense_scores[row], DENSE_POOL).indices.tolist()
        bm25_candidates = bm25.get_top_n(tokenize(query), data.corpus_ids, n=BM25_POOL)
        bm25_indices = [data.cid_to_index[cid] for cid in bm25_candidates]
        pool: List[int] = []
        seen: Set[int] = set()
        for index in dense_candidates + bm25_indices:
            if index in relevant_indices or index in seen:
                continue
            if dense_scores[row, index].item() >= best_positive_score - FALSE_NEGATIVE_MARGIN:
                continue
            seen.add(index)
            pool.append(index)
        pool.sort(key=lambda index: dense_scores[row, index].item(), reverse=True)
        chosen = pool[:NEGATIVES_PER_QUERY]
        if not chosen:
            chosen = [index for index in dense_candidates if index not in relevant_indices][:NEGATIVES_PER_QUERY]
        for positive_cid in relevant_cids:
            positive_index = data.cid_to_index[positive_cid]
            for negative_index in chosen:
                anchors.append(query)
                positives.append(data.corpus_texts[positive_index])
                negatives.append(data.corpus_texts[negative_index])
    return Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})


def train_round(
    model: SentenceTransformer,
    loss: CachedMultipleNegativesRankingLoss,
    round_index: int,
    triplet_dataset: Dataset,
) -> None:
    args = SentenceTransformerTrainingArguments(
        output_dir=f"{OUTPUT_DIR}/round_{round_index}",
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={"anchor": QUERY_PROMPT},
        logging_steps=10,
        save_strategy="no",
        report_to="wandb",
        run_name=f"vnlegal-lal-ance-round-{round_index}",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=triplet_dataset,
        loss=loss,
    )
    trainer.train()


def run_ance(model: SentenceTransformer, data: DataBundle, bm25: BM25Okapi, evaluator: InformationRetrievalEvaluator) -> None:
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=4)
    for round_index in range(1, ANCE_ROUNDS + 1):
        print(f"ANCE round {round_index}/{ANCE_ROUNDS}: encoding corpus and mining hard negatives")
        document_embeddings = encode_corpus(model, data.corpus_texts)
        triplet_dataset = mine_triplets(model, data, bm25, document_embeddings)
        print(f"round {round_index}: mined {len(triplet_dataset)} triplets")
        train_round(model, loss, round_index, triplet_dataset)
        round_metrics = evaluator(model)
        print(f"round {round_index}: ndcg@10 = {round_metrics.get('alqac_cosine_ndcg@10')}")


def merge_and_save(model: SentenceTransformer) -> None:
    model[0].model = model[0].model.merge_and_unload()
    model.save_pretrained(MERGED_DIR)


def evaluate_retrieval(model: SentenceTransformer, data: DataBundle) -> None:
    document_embeddings = model.encode(
        data.corpus_texts,
        batch_size=16,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_ids = list(data.queries.keys())
    query_texts = [data.queries[qid] for qid in query_ids]
    query_embeddings = model.encode(
        query_texts,
        prompt=QUERY_PROMPT,
        batch_size=16,
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
            retrieved = [data.corpus_ids[index] for index in ranking[row, :k]]
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
    for key in sorted(results):
        print(key, results[key])


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
            "- lora",
            "- ance",
            "- legal",
            "- vietnamese",
            f"base_model: {MODEL_NAME}",
            "---",
        ]
    )

    body = f"""
# ALQAC VNLegal-LAL Retriever

A **Vietnamese legal retrieval embedding model**, fine-tuned from
`{MODEL_NAME}` for the ALQAC legal document retrieval task. Given a legal
question it retrieves the **relevant law articles** (dense retrieval).

## Training

- **Base model**: `{MODEL_NAME}` (max_seq_length = {MAX_SEQ_LENGTH}).
- **Method**: LoRA fine-tuning (rank 16) with **ANCE** — {ANCE_ROUNDS} rounds of
  hard-negative mining, combining dense retrieval and BM25 candidates, with a
  false-negative margin filter. The LoRA adapter is merged into the base model
  before publishing.
- **Loss**: `CachedMultipleNegativesRankingLoss`.
- **Query prompt** (must be used at query time):

  ```
  {QUERY_PROMPT.strip()}
  ```

- Documents (law articles) are encoded **without** the prompt.

## Results (retrieval on the held-out ALQAC split)

{metrics_block}

## Usage

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("{HUB_MODEL_ID}")

query_prompt = (
    "Instruct: Given a Vietnamese legal question, retrieve relevant legal "
    "passages that answer the question\\nQuery: "
)

query = "How is compensation for damage caused by animals regulated?"
documents = [
    "Article 603. Compensation for damage caused by animals ...",
    "Article 584. Basis for arising liability to compensate for damage ...",
]

# Encode the query WITH the instruct prompt, documents WITHOUT it.
query_emb = model.encode(query, prompt=query_prompt, normalize_embeddings=True)
doc_emb = model.encode(documents, normalize_embeddings=True)

scores = cos_sim(query_emb, doc_emb)[0]
for doc, score in sorted(zip(documents, scores), key=lambda pair: -pair[1]):
    print(round(float(score), 4), doc[:60])
```

## Limitations

- Trained on the small-scale ALQAC 2026 dataset, tuned for the Vietnamese
  legal domain.
- The query-side instruct prompt is required to reproduce the reported
  retrieval quality.
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
    model.push_to_hub(HUB_MODEL_ID, private=True, exist_ok=True)

    readme_path = Path(MERGED_DIR) / "README.md"
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
    bm25 = build_bm25(data)
    attach_lora(model)
    run_ance(model, data, bm25, evaluator)
    merge_and_save(model)
    print(f"Đã lưu model tuned tại local: {MERGED_DIR}")
    final_metrics = evaluator(model)
    print_metrics(final_metrics)
    evaluate_retrieval(model, data)
    push_to_hub(model, final_metrics)


if __name__ == "__main__":
    main()
