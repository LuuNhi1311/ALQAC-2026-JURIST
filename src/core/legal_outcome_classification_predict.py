import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

from legal_outcome_classification_train import (
    DistillationCollator,
    VerdictDataset,
    evaluate_model,
    load_json_or_jsonl,
    load_tokenizer,
    move_inputs_to_device,
    normalize_text,
    prepare_dataframe,
    save_evaluation_results,
)


@dataclass
class PredictConfig:
    model_dir: str
    data_path: Optional[str] = None
    query: Optional[str] = None
    output_path: Optional[str] = None

    student_max_length: int = 512
    eval_batch_size: int = 4
    concat_separator: str = " [SEP] "
    trust_remote_code: bool = False
    num_workers: int = 0

    gpu_ids: List[int] = field(default_factory=list)
    device: torch.device = field(
        default_factory=lambda: torch.device("cpu")
    )

    def resolve_device(self) -> None:
        if torch.cuda.is_available() and self.gpu_ids:
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


def load_student_model(config: PredictConfig) -> Any:
    model_path = Path(config.model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {config.model_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=config.trust_remote_code,
    ).to(config.device)

    model.eval()

    return model


def load_student_tokenizer(config: PredictConfig) -> Any:
    return load_tokenizer(config.model_dir, config.trust_remote_code)


def build_id2label(model: Any) -> Dict[int, str]:
    """Read the model's stored id->label map, coercing string keys back to int class ids."""
    return {int(key): value for key, value in model.config.id2label.items()}


@torch.no_grad()
def predict_case_query(
    query: str,
    model: Any,
    tokenizer: Any,
    config: PredictConfig,
) -> Dict[str, Any]:
    """Predict a verdict for one case_query: normalize -> tokenize -> softmax -> argmax label."""
    normalized_query = normalize_text(query)

    # Guard against empty/whitespace input, which would otherwise yield a meaningless prediction.
    if not normalized_query:
        raise ValueError("case_query không được để trống.")

    encoded = tokenizer(
        normalized_query,
        padding=True,
        truncation=True,
        max_length=config.student_max_length,
        return_tensors="pt",
    )
    encoded = move_inputs_to_device(encoded, config.device)

    logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(probabilities.argmax().item())
    id2label = build_id2label(model)

    probability_by_label = {
        id2label[class_id]: float(probabilities[class_id].item())
        for class_id in sorted(id2label.keys())
    }

    return {
        "verdict_label": id2label[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "probabilities": probability_by_label,
    }


@torch.no_grad()
def predict_dataframe(
    dataframe: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    config: PredictConfig,
) -> List[Dict[str, Any]]:
    """Batch inference over student_text: softmax per batch, argmax to labels, keep full per-label probs."""
    model.eval()

    id2label = build_id2label(model)
    results: List[Dict[str, Any]] = []

    student_texts = dataframe["student_text"].tolist()

    progress_bar = tqdm(
        range(0, len(student_texts), config.eval_batch_size),
        desc="Predict student",
        colour="green",
    )

    for start in progress_bar:
        batch_texts = student_texts[
            start : start + config.eval_batch_size
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config.student_max_length,
            return_tensors="pt",
        )
        encoded = move_inputs_to_device(encoded, config.device)

        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_ids = probabilities.argmax(dim=-1)

        for row_index in range(len(batch_texts)):
            predicted_id = int(predicted_ids[row_index].item())
            row_probabilities = probabilities[row_index]

            probability_by_label = {
                id2label[class_id]: float(row_probabilities[class_id].item())
                for class_id in sorted(id2label.keys())
            }

            results.append(
                {
                    "verdict_label": id2label[predicted_id],
                    "confidence": float(
                        row_probabilities[predicted_id].item()
                    ),
                    "probabilities": probability_by_label,
                }
            )

    return results


def attach_reference_columns(
    dataframe: pd.DataFrame,
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []

    for row_index, prediction in enumerate(predictions):
        row = dataframe.iloc[row_index]
        record = dict(prediction)

        if "case_id" in dataframe.columns:
            record["case_id"] = row["case_id"]

        record["case_query"] = row["student_text"]

        if "verdict_label" in dataframe.columns:
            record["true_verdict_label"] = row["verdict_label"]

        enriched.append(record)

    return enriched


def evaluate_predictions(
    dataframe: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    config: PredictConfig,
    output_directory: Path,
) -> None:
    """Score predictions against gold labels; invert id2label so the dataset can encode labels back to ids."""
    id2label = build_id2label(model)
    label2id = {value: key for key, value in id2label.items()}

    dataset = VerdictDataset(dataframe, label2id)

    collator = DistillationCollator(
        teacher_tokenizer=tokenizer,
        student_tokenizer=tokenizer,
        teacher_max_length=config.student_max_length,
        student_max_length=config.student_max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    metrics, labels, predictions = evaluate_model(
        model,
        dataloader,
        input_type="student",
        id2label=id2label,
        device=config.device,
    )

    save_evaluation_results(
        metrics=metrics,
        labels=labels,
        predictions=predictions,
        id2label=id2label,
        output_directory=output_directory,
        prefix="predict",
    )

    print(
        "\nĐánh giá trên dữ liệu có nhãn:"
        f"\n  Macro F1: {metrics['macro_f1']:.4f}"
        f"\n  Weighted F1: {metrics['weighted_f1']:.4f}"
        f"\n  Balanced accuracy: {metrics['balanced_accuracy']:.4f}"
        f"\n  Accuracy: {metrics['accuracy']:.4f}"
    )


def run_single_query(config: PredictConfig) -> None:
    model = load_student_model(config)
    tokenizer = load_student_tokenizer(config)

    result = predict_case_query(config.query, model, tokenizer, config)

    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_batch_prediction(config: PredictConfig) -> None:
    model = load_student_model(config)
    tokenizer = load_student_tokenizer(config)

    raw_dataframe = load_json_or_jsonl(config.data_path)
    dataframe = prepare_dataframe(raw_dataframe, config.concat_separator)

    print(f"Số mẫu dự đoán: {len(dataframe)}")

    predictions = predict_dataframe(dataframe, model, tokenizer, config)
    enriched = attach_reference_columns(dataframe, predictions)

    if config.output_path:
        output_file = Path(config.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(enriched, file, ensure_ascii=False, indent=2)

        print(f"Đã lưu dự đoán tại: {output_file}")
    else:
        print(json.dumps(enriched, ensure_ascii=False, indent=2))

    if "verdict_label" in dataframe.columns:
        output_directory = (
            Path(config.output_path).parent
            if config.output_path
            else Path(config.model_dir)
        )
        evaluate_predictions(
            dataframe, model, tokenizer, config, output_directory
        )


def run_prediction(config: PredictConfig) -> None:
    config.resolve_device()

    print(f"Device: {config.device}")
    print(f"GPU ids: {config.gpu_ids}")

    if config.query:
        run_single_query(config)
    elif config.data_path:
        run_batch_prediction(config)
    else:
        raise ValueError("Cần cung cấp --query hoặc --data để dự đoán.")


def parse_gpu_ids(raw_value: Optional[str]) -> List[int]:
    if not raw_value:
        return []

    ids: List[int] = []

    for token in raw_value.replace(" ", ",").split(","):
        token = token.strip()

        if token:
            ids.append(int(token))

    return ids


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference student chỉ với case_query."
    )

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--student-max-length", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Danh sách GPU, ví dụ '0' hoặc '0,1'.",
    )

    return parser


def build_config_from_args(
    arguments: argparse.Namespace,
) -> PredictConfig:
    return PredictConfig(
        model_dir=arguments.model_dir,
        data_path=arguments.data,
        query=arguments.query,
        output_path=arguments.output,
        student_max_length=arguments.student_max_length,
        eval_batch_size=arguments.eval_batch_size,
        num_workers=arguments.num_workers,
        trust_remote_code=arguments.trust_remote_code,
        gpu_ids=parse_gpu_ids(arguments.gpu_ids),
    )


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    config = build_config_from_args(arguments)

    run_prediction(config)


if __name__ == "__main__":
    main()
