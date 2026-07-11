import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

try:
    import wandb
except ImportError:
    wandb = None


DEFAULT_TEACHER_MODEL_NAME: str = "Qualcomm-AI-Research/BamiBERT"
DEFAULT_STUDENT_MODEL_NAME: str = "Qualcomm-AI-Research/BamiBERT"

HUB_MODEL_ID: str = "leonpham1208/alqac_legal_outcome_cls"

REQUIRED_COLUMNS: List[str] = [
    "court",
    "court_level",
    "court_verdict",
    "judgment_text",
    "judgment_number",
    "related_law_provisions",
    "A_role",
    "A_description",
    "B_role",
    "B_description",
    "case_query",
    "court_reasoning",
    "verdict_label",
]

COLOR_GREEN: str = "\033[92m"
COLOR_YELLOW: str = "\033[93m"
COLOR_RED: str = "\033[91m"
COLOR_CYAN: str = "\033[96m"
COLOR_RESET: str = "\033[0m"

LOSS_COLOR_MAP: Dict[str, str] = {
    "total": COLOR_GREEN,
    "hard": COLOR_YELLOW,
    "soft": COLOR_RED,
    "rep": COLOR_CYAN,
    "loss": COLOR_GREEN,
    "lr": COLOR_CYAN,
}

_WANDB_ACTIVE: bool = False
_GLOBAL_STEP: int = 0


def reset_global_step() -> None:
    global _GLOBAL_STEP
    _GLOBAL_STEP = 0


def increment_global_step() -> int:
    global _GLOBAL_STEP
    _GLOBAL_STEP += 1
    return _GLOBAL_STEP


@dataclass
class TrainingConfig:
    data_path: Optional[str] = None
    train_data_path: Optional[str] = None
    valid_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    output_dir: str = "./verdict_outputs"

    teacher_model_name: str = DEFAULT_TEACHER_MODEL_NAME
    student_model_name: str = DEFAULT_STUDENT_MODEL_NAME
    trust_remote_code: bool = False

    seed: int = 42

    teacher_max_length: int = 2048
    student_max_length: int = 512

    teacher_epochs: int = 5
    student_epochs: int = 8

    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    teacher_learning_rate: float = 2e-5
    student_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05
    lr_scheduler_type: str = "cosine"

    distill_ce_weight: float = 1.0
    distill_alpha: float = 0.60
    distill_beta: float = 0.30
    distill_temperature: float = 2.0

    early_stopping_patience: int = 3

    freeze_encoder_layers: int = 0
    freeze_embeddings: bool = False

    kfold: int = 0

    split_mode: str = "random"
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    test_ratio: float = 0.15

    concat_separator: str = " [SEP] "
    num_workers: int = 0

    gpu_ids: List[int] = field(default_factory=list)
    device: torch.device = field(
        default_factory=lambda: torch.device("cpu")
    )
    use_mixed_precision: bool = False

    use_wandb: bool = True
    wandb_project: str = "alqac-legal-outcome-distillation"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    push_to_hub: bool = True
    hub_model_id: str = HUB_MODEL_ID
    hub_private: bool = True
    hub_license: str = "apache-2.0"

    def resolve_runtime(self) -> None:
        if torch.cuda.is_available() and self.gpu_ids:
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            self.use_mixed_precision = True
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.use_mixed_precision = True
        else:
            self.device = torch.device("cpu")
            self.use_mixed_precision = False

    def to_serializable_dict(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
    ) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "teacher_model_name": self.teacher_model_name,
            "student_model_name": self.student_model_name,
            "teacher_max_length": self.teacher_max_length,
            "student_max_length": self.student_max_length,
            "teacher_epochs": self.teacher_epochs,
            "student_epochs": self.student_epochs,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "teacher_learning_rate": self.teacher_learning_rate,
            "student_learning_rate": self.student_learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "label_smoothing": self.label_smoothing,
            "lr_scheduler_type": self.lr_scheduler_type,
            "distill_ce_weight": self.distill_ce_weight,
            "distill_alpha": self.distill_alpha,
            "distill_beta": self.distill_beta,
            "distill_temperature": self.distill_temperature,
            "loss_formula": (
                "ce_weight*CE + alpha*KL + beta*representation_alignment"
            ),
            "metric_for_best_model": "macro_f1",
            "early_stopping_patience": self.early_stopping_patience,
            "freeze_encoder_layers": self.freeze_encoder_layers,
            "freeze_embeddings": self.freeze_embeddings,
            "kfold": self.kfold,
            "split_mode": self.split_mode,
            "gpu_ids": self.gpu_ids,
            "device": str(self.device),
            "use_mixed_precision": self.use_mixed_precision,
            "teacher_features": [
                "court + court_level",
                "court_verdict",
                "judgment_text",
                "judgment_number",
                "related_law_provisions",
                "A_role + A_description",
                "B_role + B_description",
                "case_query",
                "court + court_level + court_reasoning",
            ],
            "student_features": ["case_query"],
            "label2id": label2id,
            "id2label": {
                str(key): value for key, value in id2label.items()
            },
        }


def colorize(color: str, text: str) -> str:
    return f"{color}{text}{COLOR_RESET}"


def format_loss_postfix(**named_losses: float) -> str:
    parts: List[str] = []

    for name, value in named_losses.items():
        color = LOSS_COLOR_MAP.get(name, COLOR_GREEN)
        parts.append(colorize(color, f"{name}={value:.4f}"))

    return " ".join(parts)


def wandb_log(
    payload: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    if not _WANDB_ACTIVE or wandb is None:
        return

    if step is not None:
        wandb.log(payload, step=step)
    else:
        wandb.log(payload)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_or_jsonl(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with path.open("r", encoding="utf-8") as file:
        raw_text = file.read().strip()

    if not raw_text:
        raise ValueError("File dữ liệu đang rỗng.")

    try:
        data = json.loads(raw_text)

        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("JSON phải là object hoặc danh sách object.")

    except json.JSONDecodeError:
        records = []

        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL lỗi tại dòng {line_number}: {error}"
                ) from error

    dataframe = pd.DataFrame(records)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Dữ liệu thiếu các cột: {missing_columns}")

    return dataframe


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value)
    text = " ".join(text.split())

    return text.strip()


def concat_string(separator: str, *values: Any) -> str:
    normalized_values = [normalize_text(value) for value in values]
    normalized_values = [value for value in normalized_values if value]

    return separator.join(normalized_values)


def add_section(
    sections: List[str],
    section_name: str,
    section_text: str,
) -> None:
    section_text = normalize_text(section_text)

    if section_text:
        sections.append(f"[{section_name}] {section_text}")


def build_teacher_text(row: pd.Series, separator: str) -> str:
    court_and_level = concat_string(
        separator, row["court"], row["court_level"]
    )

    party_a = concat_string(
        separator, row["A_role"], row["A_description"]
    )

    party_b = concat_string(
        separator, row["B_role"], row["B_description"]
    )

    court_level_reasoning = concat_string(
        separator, row["court"], row["court_level"], row["court_reasoning"]
    )

    sections: List[str] = []

    add_section(sections, "CASE_QUERY", row["case_query"])
    add_section(sections, "COURT_AND_LEVEL", court_and_level)
    add_section(sections, "PARTY_A", party_a)
    add_section(sections, "PARTY_B", party_b)
    add_section(sections, "JUDGMENT_NUMBER", row["judgment_number"])
    add_section(
        sections, "RELATED_LAW_PROVISIONS", row["related_law_provisions"]
    )
    add_section(sections, "COURT_LEVEL_REASONING", court_level_reasoning)
    add_section(sections, "COURT_VERDICT", row["court_verdict"])
    add_section(sections, "JUDGMENT_TEXT", row["judgment_text"])

    return "\n".join(sections)


def prepare_dataframe(
    dataframe: pd.DataFrame,
    separator: str,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    for column in REQUIRED_COLUMNS:
        dataframe[column] = dataframe[column].apply(normalize_text)

    dataframe["teacher_text"] = dataframe.apply(
        lambda row: build_teacher_text(row, separator),
        axis=1,
    )

    dataframe["student_text"] = dataframe["case_query"]

    valid_mask = (
        dataframe["teacher_text"].str.len().gt(0)
        & dataframe["student_text"].str.len().gt(0)
        & dataframe["verdict_label"].str.len().gt(0)
    )

    dataframe = dataframe[valid_mask].reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("Không còn mẫu hợp lệ sau tiền xử lý.")

    return dataframe


def create_label_mapping(
    dataframe: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(
        dataframe["verdict_label"].astype(str).unique().tolist()
    )

    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    return label2id, id2label


def random_split_dataframe(
    dataframe: pd.DataFrame,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_counts = dataframe["verdict_label"].value_counts()

    can_stratify = len(label_counts) > 1 and label_counts.min() >= 3
    stratify_labels = (
        dataframe["verdict_label"] if can_stratify else None
    )

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    relative_test_ratio = test_ratio / (valid_ratio + test_ratio)

    temp_counts = temp_df["verdict_label"].value_counts()

    can_stratify_temp = len(temp_counts) > 1 and temp_counts.min() >= 2
    temp_stratify = (
        temp_df["verdict_label"] if can_stratify_temp else None
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_stratify,
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def time_split_dataframe(
    dataframe: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "judgment_date" not in dataframe.columns:
        raise ValueError("Time split yêu cầu cột judgment_date.")

    dataframe = dataframe.copy()

    dataframe["_parsed_date"] = pd.to_datetime(
        dataframe["judgment_date"],
        errors="coerce",
        dayfirst=True,
    )

    missing_date_count = dataframe["_parsed_date"].isna().sum()

    if missing_date_count > 0:
        raise ValueError(
            f"Có {missing_date_count} mẫu không đọc được ngày."
        )

    dataframe = dataframe.sort_values("_parsed_date").reset_index(drop=True)

    total_size = len(dataframe)

    train_end = int(total_size * train_ratio)
    valid_end = int(total_size * (train_ratio + valid_ratio))

    train_df = dataframe.iloc[:train_end].drop(columns=["_parsed_date"])
    valid_df = dataframe.iloc[train_end:valid_end].drop(
        columns=["_parsed_date"]
    )
    test_df = dataframe.iloc[valid_end:].drop(columns=["_parsed_date"])

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_dataframe(
    dataframe: pd.DataFrame,
    config: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.split_mode == "time":
        return time_split_dataframe(
            dataframe,
            train_ratio=config.train_ratio,
            valid_ratio=config.valid_ratio,
        )

    return random_split_dataframe(
        dataframe,
        seed=config.seed,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
    )


class VerdictDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        label2id: Dict[str, int],
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[index]
        label_name = str(row["verdict_label"])

        if label_name not in self.label2id:
            raise KeyError(f"Nhãn chưa được khai báo: {label_name}")

        return {
            "teacher_text": row["teacher_text"],
            "student_text": row["student_text"],
            "label": self.label2id[label_name],
        }


class DistillationCollator:
    def __init__(
        self,
        teacher_tokenizer: Any,
        student_tokenizer: Any,
        teacher_max_length: int,
        student_max_length: int,
    ) -> None:
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.teacher_max_length = teacher_max_length
        self.student_max_length = student_max_length

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        teacher_texts = [item["teacher_text"] for item in batch]
        student_texts = [item["student_text"] for item in batch]

        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        )

        teacher_inputs = self.teacher_tokenizer(
            teacher_texts,
            padding=True,
            truncation=True,
            max_length=self.teacher_max_length,
            return_tensors="pt",
        )

        student_inputs = self.student_tokenizer(
            student_texts,
            padding=True,
            truncation=True,
            max_length=self.student_max_length,
            return_tensors="pt",
        )

        return {
            "teacher_inputs": teacher_inputs,
            "student_inputs": student_inputs,
            "labels": labels,
        }


def calculate_class_weights(
    train_dataframe: pd.DataFrame,
    label2id: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    label_ids = [
        label2id[label]
        for label in train_dataframe["verdict_label"].astype(str)
    ]

    num_labels = len(label2id)
    counts = np.bincount(label_ids, minlength=num_labels)

    missing_classes = np.where(counts == 0)[0]

    if len(missing_classes) > 0:
        missing_names = [str(class_id) for class_id in missing_classes]
        raise ValueError(
            "Train split thiếu các class id: " + ", ".join(missing_names)
        )

    weights = len(label_ids) / (num_labels * counts)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool,
) -> Any:
    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except (TypeError, ValueError):
        return PreTrainedTokenizerFast.from_pretrained(model_name_or_path)


def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    scheduler_type: str,
):
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def move_inputs_to_device(
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def enable_gradient_checkpointing(model: nn.Module) -> None:
    try:
        model.gradient_checkpointing_enable()
    except (AttributeError, ValueError):
        pass

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module

    return model


def wrap_data_parallel(
    model: nn.Module,
    gpu_ids: List[int],
) -> nn.Module:
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        return nn.DataParallel(model, device_ids=gpu_ids)

    return model


def freeze_encoder(
    model: nn.Module,
    num_layers: int,
    freeze_embeddings: bool,
) -> None:
    if num_layers <= 0 and not freeze_embeddings:
        return

    base = model.base_model

    if freeze_embeddings and hasattr(base, "embeddings"):
        for parameter in base.embeddings.parameters():
            parameter.requires_grad = False

    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None) if encoder is not None else None

    if layers is not None and num_layers > 0:
        for layer in layers[:num_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    frozen = sum(
        1 for parameter in model.parameters() if not parameter.requires_grad
    )
    total = sum(1 for _ in model.parameters())
    print(f"Freeze: {frozen}/{total} tensor bị đóng băng.")


def load_classification_model(
    model_name: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    config: TrainingConfig,
) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        trust_remote_code=config.trust_remote_code,
    )

    freeze_encoder(
        model,
        config.freeze_encoder_layers,
        config.freeze_embeddings,
    )

    enable_gradient_checkpointing(model)
    model = model.to(config.device)

    return wrap_data_parallel(model, config.gpu_ids)


def load_pretrained_model(
    model_directory: Path,
    config: TrainingConfig,
) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_directory,
        trust_remote_code=config.trust_remote_code,
    ).to(config.device)

    return wrap_data_parallel(model, config.gpu_ids)


def get_pooled_representation(outputs: Any) -> torch.Tensor:
    last_hidden_state = outputs.hidden_states[-1]

    return last_hidden_state[:, 0]


class RepresentationProjector(nn.Module):
    def __init__(
        self,
        student_hidden_size: int,
        teacher_hidden_size: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(
            student_hidden_size,
            teacher_hidden_size,
        )

    def forward(self, student_hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(student_hidden)


def representation_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    projector: RepresentationProjector,
) -> torch.Tensor:
    projected_student = projector(student_hidden)
    projected_student = F.normalize(projected_student, dim=-1)
    teacher_normalized = F.normalize(teacher_hidden, dim=-1)

    cosine_similarity = (projected_student * teacher_normalized).sum(dim=-1)

    return (1.0 - cosine_similarity).mean()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    input_type: str,
    id2label: Dict[int, str],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[int], List[int]]:
    model.eval()

    all_predictions: List[int] = []
    all_labels: List[int] = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Evaluate {input_type}",
        colour="cyan",
    )

    for batch in progress_bar:
        labels = batch["labels"].to(device)

        if input_type == "teacher":
            inputs = move_inputs_to_device(batch["teacher_inputs"], device)
        elif input_type == "student":
            inputs = move_inputs_to_device(batch["student_inputs"], device)
        else:
            raise ValueError("input_type phải là teacher hoặc student.")

        logits = model(**inputs).logits
        predictions = logits.argmax(dim=-1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    label_ids = sorted(id2label.keys())
    target_names = [id2label[label_id] for label_id in label_ids]

    report = classification_report(
        all_labels,
        all_predictions,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(all_labels, all_predictions)
        ),
        "macro_precision": float(
            precision_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": report,
    }

    return metrics, all_labels, all_predictions


def save_evaluation_results(
    metrics: Dict[str, Any],
    labels: List[int],
    predictions: List[int],
    id2label: Dict[int, str],
    output_directory: Path,
    prefix: str,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    metrics_path = output_directory / f"{prefix}_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.item() if hasattr(value, "item") else str(value)
            ),
        )

    label_ids = sorted(id2label.keys())

    matrix = confusion_matrix(labels, predictions, labels=label_ids)
    label_names = [id2label[label_id] for label_id in label_ids]

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=label_names,
        columns=label_names,
    )

    matrix_dataframe.to_csv(
        output_directory / f"{prefix}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )


def train_teacher(
    teacher: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    class_weights: torch.Tensor,
    id2label: Dict[int, str],
    output_directory: Path,
    config: TrainingConfig,
) -> Path:
    teacher_output = output_directory / "teacher_best"

    optimizer = AdamW(
        teacher.parameters(),
        lr=config.teacher_learning_rate,
        weight_decay=config.weight_decay,
    )

    updates_per_epoch = int(
        np.ceil(len(train_loader) / config.gradient_accumulation_steps)
    )
    total_training_steps = updates_per_epoch * config.teacher_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        scheduler_type=config.lr_scheduler_type,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.use_mixed_precision)

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(config.teacher_epochs):
        teacher.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Teacher epoch {epoch + 1}/{config.teacher_epochs}",
            colour="yellow",
        )

        for step, batch in progress_bar:
            labels = batch["labels"].to(config.device)
            teacher_inputs = move_inputs_to_device(
                batch["teacher_inputs"], config.device
            )

            with torch.amp.autocast("cuda", enabled=config.use_mixed_precision):
                logits = teacher(**teacher_inputs).logits

                loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                    label_smoothing=config.label_smoothing,
                )

                scaled_loss = loss / config.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            running_loss += loss.item()

            should_update = (
                (step + 1) % config.gradient_accumulation_steps == 0
            )
            is_last_step = step + 1 == len(train_loader)

            if should_update or is_last_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    teacher.parameters(), config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            global_step = increment_global_step()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix_str(
                format_loss_postfix(loss=loss.item(), lr=current_lr)
            )

            wandb_log(
                {
                    "teacher/train_loss": loss.item(),
                    "teacher/learning_rate": current_lr,
                    "teacher/epoch": epoch + 1,
                },
                step=global_step,
            )

        average_loss = running_loss / max(len(train_loader), 1)

        valid_metrics, valid_labels, valid_predictions = evaluate_model(
            teacher,
            valid_loader,
            input_type="teacher",
            id2label=id2label,
            device=config.device,
        )

        macro_f1 = valid_metrics["macro_f1"]

        print(
            f"\nTeacher epoch {epoch + 1}: "
            f"loss={average_loss:.4f}, "
            f"macro_f1={macro_f1:.4f}, "
            f"accuracy={valid_metrics['accuracy']:.4f}"
        )

        wandb_log(
            {
                "teacher/train_loss_epoch": average_loss,
                "teacher/val_macro_f1": macro_f1,
                "teacher/val_accuracy": valid_metrics["accuracy"],
                "teacher/val_weighted_f1": valid_metrics["weighted_f1"],
                "teacher/val_balanced_accuracy": valid_metrics[
                    "balanced_accuracy"
                ],
                "teacher/epoch": epoch + 1,
            },
            step=global_step,
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0

            teacher_output.mkdir(parents=True, exist_ok=True)
            unwrap_model(teacher).save_pretrained(teacher_output)

            save_evaluation_results(
                metrics=valid_metrics,
                labels=valid_labels,
                predictions=valid_predictions,
                id2label=id2label,
                output_directory=teacher_output,
                prefix="validation",
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "Teacher dừng sớm vì validation không còn cải thiện."
                )
                break

    wandb_log({"teacher/best_val_macro_f1": best_macro_f1})

    return teacher_output


def train_student(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    class_weights: torch.Tensor,
    id2label: Dict[int, str],
    student_tokenizer: Any,
    output_directory: Path,
    config: TrainingConfig,
) -> Path:
    student_output = output_directory / "student_query_only"

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    teacher.eval()

    projector = RepresentationProjector(
        student_hidden_size=unwrap_model(student).config.hidden_size,
        teacher_hidden_size=unwrap_model(teacher).config.hidden_size,
    ).to(config.device)

    trainable_parameters = list(student.parameters()) + list(
        projector.parameters()
    )

    optimizer = AdamW(
        trainable_parameters,
        lr=config.student_learning_rate,
        weight_decay=config.weight_decay,
    )

    updates_per_epoch = int(
        np.ceil(len(train_loader) / config.gradient_accumulation_steps)
    )
    total_training_steps = updates_per_epoch * config.student_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        scheduler_type=config.lr_scheduler_type,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.use_mixed_precision)

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    global_step = 0

    temperature = config.distill_temperature
    ce_weight = config.distill_ce_weight
    alpha = config.distill_alpha
    beta = config.distill_beta

    for epoch in range(config.student_epochs):
        student.train()
        projector.train()
        optimizer.zero_grad(set_to_none=True)

        running_total_loss = 0.0
        running_hard_loss = 0.0
        running_soft_loss = 0.0
        running_rep_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Student epoch {epoch + 1}/{config.student_epochs}",
            colour="green",
        )

        for step, batch in progress_bar:
            labels = batch["labels"].to(config.device)
            teacher_inputs = move_inputs_to_device(
                batch["teacher_inputs"], config.device
            )
            student_inputs = move_inputs_to_device(
                batch["student_inputs"], config.device
            )

            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda", enabled=config.use_mixed_precision
                ):
                    teacher_outputs = teacher(
                        **teacher_inputs,
                        output_hidden_states=True,
                    )

                teacher_logits = teacher_outputs.logits.float()
                teacher_hidden = get_pooled_representation(
                    teacher_outputs
                ).float()

            with torch.amp.autocast("cuda", enabled=config.use_mixed_precision):
                student_outputs = student(
                    **student_inputs,
                    output_hidden_states=True,
                )

                student_logits = student_outputs.logits
                student_hidden = get_pooled_representation(student_outputs)

                hard_loss = F.cross_entropy(
                    student_logits,
                    labels,
                    weight=class_weights,
                    label_smoothing=config.label_smoothing,
                )

                soft_loss = F.kl_div(
                    F.log_softmax(student_logits.float() / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * (temperature ** 2)

                rep_loss = representation_alignment_loss(
                    student_hidden=student_hidden.float(),
                    teacher_hidden=teacher_hidden,
                    projector=projector,
                )

                total_loss = (
                    ce_weight * hard_loss
                    + alpha * soft_loss
                    + beta * rep_loss
                )

                scaled_loss = total_loss / config.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            running_total_loss += total_loss.item()
            running_hard_loss += hard_loss.item()
            running_soft_loss += soft_loss.item()
            running_rep_loss += rep_loss.item()

            should_update = (
                (step + 1) % config.gradient_accumulation_steps == 0
            )
            is_last_step = step + 1 == len(train_loader)

            if should_update or is_last_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            global_step = increment_global_step()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix_str(
                format_loss_postfix(
                    total=total_loss.item(),
                    hard=hard_loss.item(),
                    soft=soft_loss.item(),
                    rep=rep_loss.item(),
                )
            )

            wandb_log(
                {
                    "student/total_loss": total_loss.item(),
                    "student/hard_loss_ce": hard_loss.item(),
                    "student/soft_loss_kl": soft_loss.item(),
                    "student/rep_align_loss": rep_loss.item(),
                    "student/learning_rate": current_lr,
                    "student/epoch": epoch + 1,
                },
                step=global_step,
            )

        steps = max(len(train_loader), 1)
        average_total_loss = running_total_loss / steps
        average_hard_loss = running_hard_loss / steps
        average_soft_loss = running_soft_loss / steps
        average_rep_loss = running_rep_loss / steps

        valid_metrics, valid_labels, valid_predictions = evaluate_model(
            student,
            valid_loader,
            input_type="student",
            id2label=id2label,
            device=config.device,
        )

        macro_f1 = valid_metrics["macro_f1"]

        print(
            f"\nStudent epoch {epoch + 1}: "
            f"total={average_total_loss:.4f}, "
            f"hard={average_hard_loss:.4f}, "
            f"soft={average_soft_loss:.4f}, "
            f"rep={average_rep_loss:.4f}, "
            f"macro_f1={macro_f1:.4f}"
        )

        wandb_log(
            {
                "student/total_loss_epoch": average_total_loss,
                "student/hard_loss_epoch": average_hard_loss,
                "student/soft_loss_epoch": average_soft_loss,
                "student/rep_loss_epoch": average_rep_loss,
                "student/val_macro_f1": macro_f1,
                "student/val_accuracy": valid_metrics["accuracy"],
                "student/val_weighted_f1": valid_metrics["weighted_f1"],
                "student/val_balanced_accuracy": valid_metrics[
                    "balanced_accuracy"
                ],
                "student/epoch": epoch + 1,
            },
            step=global_step,
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0

            student_output.mkdir(parents=True, exist_ok=True)
            unwrap_model(student).save_pretrained(student_output)
            student_tokenizer.save_pretrained(student_output)

            save_evaluation_results(
                metrics=valid_metrics,
                labels=valid_labels,
                predictions=valid_predictions,
                id2label=id2label,
                output_directory=student_output,
                prefix="validation",
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "Student dừng sớm vì validation không còn cải thiện."
                )
                break

    wandb_log({"student/best_val_macro_f1": best_macro_f1})

    return student_output


def save_training_configuration(
    output_directory: Path,
    config: TrainingConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)

    configuration = config.to_serializable_dict(label2id, id2label)

    with (output_directory / "training_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(configuration, file, ensure_ascii=False, indent=2)

    return configuration


def initialize_wandb(
    config: TrainingConfig,
    configuration: Dict[str, Any],
) -> None:
    global _WANDB_ACTIVE

    _WANDB_ACTIVE = False

    if not config.use_wandb:
        return

    if wandb is None:
        print("Chưa cài wandb (pip install wandb), bỏ qua logging.")
        return

    try:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=configuration,
        )
        _WANDB_ACTIVE = True
        print("wandb đã được bật.")
    except Exception as error:
        print(f"Không khởi tạo được wandb, tiếp tục không log: {error}")
        _WANDB_ACTIVE = False


def finish_wandb() -> None:
    global _WANDB_ACTIVE

    if _WANDB_ACTIVE and wandb is not None:
        wandb.finish()

    _WANDB_ACTIVE = False


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


def read_metrics_summary(student_output: Path) -> Dict[str, float]:
    for prefix in ("test", "validation"):
        metrics_path = student_output / f"{prefix}_metrics.json"
        if not metrics_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        keys = (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
        )
        return {
            "split": prefix,
            **{key: metrics[key] for key in keys if key in metrics},
        }

    return {}


def build_model_card(
    config: TrainingConfig,
    id2label: Dict[int, str],
    metrics: Dict[str, Any],
) -> str:
    labels = [id2label[key] for key in sorted(id2label.keys())]

    metrics_lines = ["No evaluation results were recorded."]
    if metrics:
        split = metrics.get("split", "test")
        rows = ["| Metric | Value |", "| --- | --- |"]
        for key, value in metrics.items():
            if key == "split":
                continue
            rows.append(f"| {key} | {value:.4f} |")
        metrics_lines = [f"Evaluated on the **{split}** split:", "", *rows]

    metrics_block = "\n".join(metrics_lines)
    labels_block = "\n".join(f"- `{label}`" for label in labels)

    frontmatter = "\n".join(
        [
            "---",
            "language:",
            "- vi",
            f"license: {config.hub_license}",
            "library_name: transformers",
            "pipeline_tag: text-classification",
            "tags:",
            "- legal",
            "- vietnamese",
            "- knowledge-distillation",
            "- text-classification",
            f"base_model: {config.student_model_name}",
            "---",
        ]
    )

    body = f"""
# ALQAC Legal Outcome Classification (Student)

A model that classifies the **outcome of Vietnamese civil court cases** using
only the `case_query` (the plaintiff's short dispute summary). It is a
**student** trained by **knowledge distillation** from a teacher that reads the
full case record.

## Distillation setup

- **Teacher** (`{config.teacher_model_name}`, max_length={config.teacher_max_length}):
  reads all fields — `court + court_level + court_verdict`, `judgment_text`,
  `judgment_number`, `related_law_provisions`, `A_role + A_description`,
  `B_role + B_description`, `case_query`, `court + court_level + court_reasoning`.
- **Student** (`{config.student_model_name}`, max_length={config.student_max_length}):
  takes only `case_query`, so it can run before any judgment exists.

Student loss:

```
L = {config.distill_ce_weight}*CrossEntropy(student, label)
  + {config.distill_alpha}*KL(student, teacher)   # temperature = {config.distill_temperature}
  + {config.distill_beta}*representation_alignment(student, teacher)
```

The best checkpoint is selected by **Macro F1** on the validation split.

## Labels

{labels_block}

## Results

{metrics_block}

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

model_id = "{config.hub_model_id}"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)

case_query = "The plaintiff sues the defendant seeking compensation for damages..."
inputs = tokenizer(
    case_query,
    truncation=True,
    max_length={config.student_max_length},
    return_tensors="pt",
)

with torch.no_grad():
    probs = model(**inputs).logits.softmax(dim=-1)[0]

predicted_id = int(probs.argmax())
print(model.config.id2label[predicted_id], float(probs[predicted_id]))
```

## Limitations

- Trained on the small-scale ALQAC 2026 dataset, so it may not generalize to
  every type of dispute.
- Uses only `case_query`; it is not a substitute for human legal assessment.
"""

    return frontmatter + "\n" + body.strip() + "\n"


def push_student_to_hub(
    config: TrainingConfig,
    student_output: Path,
) -> None:
    if not config.push_to_hub:
        print("Đã tắt push (--no-push), bỏ qua đẩy lên hub.")
        return

    token = load_hf_token()
    if not token:
        print("Không tìm thấy HF_TOKEN trong .env, bỏ qua push lên hub.")
        return

    from huggingface_hub import login, upload_file

    login(token=token)

    model = AutoModelForSequenceClassification.from_pretrained(
        student_output,
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer = load_tokenizer(
        str(student_output),
        config.trust_remote_code,
    )

    id2label = {int(key): value for key, value in model.config.id2label.items()}
    metrics = read_metrics_summary(student_output)
    model_card = build_model_card(config, id2label, metrics)

    readme_path = student_output / "README.md"
    readme_path.write_text(model_card, encoding="utf-8")

    model.push_to_hub(config.hub_model_id, private=config.hub_private)
    tokenizer.push_to_hub(config.hub_model_id, private=config.hub_private)

    upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=config.hub_model_id,
        repo_type="model",
    )

    print(f"Đã push student model + model card lên hub: {config.hub_model_id}")


def load_prepared_dataframe(
    file_path: str,
    separator: str,
) -> pd.DataFrame:
    raw_dataframe = load_json_or_jsonl(file_path)

    return prepare_dataframe(raw_dataframe, separator)


def resolve_splits(
    config: TrainingConfig,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Dict[str, int],
    Dict[int, str],
]:
    if config.train_data_path and config.valid_data_path:
        train_df = load_prepared_dataframe(
            config.train_data_path, config.concat_separator
        )
        valid_df = load_prepared_dataframe(
            config.valid_data_path, config.concat_separator
        )
        test_df = (
            load_prepared_dataframe(
                config.test_data_path, config.concat_separator
            )
            if config.test_data_path
            else None
        )

        frames = [train_df, valid_df]
        if test_df is not None:
            frames.append(test_df)

        combined = pd.concat(frames, ignore_index=True)
        label2id, id2label = create_label_mapping(combined)

        return train_df, valid_df, test_df, label2id, id2label

    if config.data_path:
        dataframe = load_prepared_dataframe(
            config.data_path, config.concat_separator
        )
        label2id, id2label = create_label_mapping(dataframe)
        train_df, valid_df, test_df = split_dataframe(dataframe, config)

        return train_df, valid_df, test_df, label2id, id2label

    raise ValueError(
        "Cần --data để tự chia, hoặc --train-data và --valid-data."
    )


def train_and_eval_split(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    output_directory: Path,
    teacher_tokenizer: Any,
    student_tokenizer: Any,
) -> Dict[str, Any]:
    collator = DistillationCollator(
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        teacher_max_length=config.teacher_max_length,
        student_max_length=config.student_max_length,
    )

    train_loader = DataLoader(
        VerdictDataset(train_df, label2id),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        VerdictDataset(valid_df, label2id),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = calculate_class_weights(train_df, label2id, config.device)
    num_labels = len(label2id)

    teacher = load_classification_model(
        model_name=config.teacher_model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        config=config,
    )
    teacher_output = train_teacher(
        teacher=teacher,
        train_loader=train_loader,
        valid_loader=valid_loader,
        class_weights=class_weights,
        id2label=id2label,
        output_directory=output_directory,
        config=config,
    )
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher = load_pretrained_model(teacher_output, config)
    teacher.eval()

    student = load_classification_model(
        model_name=config.student_model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        config=config,
    )
    student_output = train_student(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        valid_loader=valid_loader,
        class_weights=class_weights,
        id2label=id2label,
        student_tokenizer=student_tokenizer,
        output_directory=output_directory,
        config=config,
    )
    del student, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return read_metrics_summary(student_output)


def run_cross_validation(
    config: TrainingConfig,
    dataframe: pd.DataFrame,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    output_directory: Path,
    teacher_tokenizer: Any,
    student_tokenizer: Any,
) -> Dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    labels = dataframe["verdict_label"].astype(str).values
    min_class_count = int(dataframe["verdict_label"].value_counts().min())
    n_splits = min(config.kfold, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "Không đủ mẫu để chạy k-fold (mỗi lớp cần >= 2 mẫu)."
        )

    print(
        f"\n===== K-FOLD CROSS-VALIDATION: {n_splits} folds "
        f"trên {len(dataframe)} mẫu ====="
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.seed,
    )

    metric_keys = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
    )
    fold_results: List[Dict[str, Any]] = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(dataframe, labels), start=1
    ):
        fold_train = dataframe.iloc[train_index].reset_index(drop=True)
        fold_valid = dataframe.iloc[valid_index].reset_index(drop=True)

        print(
            f"\n----- Fold {fold}/{n_splits} | "
            f"train={len(fold_train)} val={len(fold_valid)} -----"
        )

        metrics = train_and_eval_split(
            config=config,
            train_df=fold_train,
            valid_df=fold_valid,
            label2id=label2id,
            id2label=id2label,
            output_directory=output_directory / "cv" / f"fold_{fold}",
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
        )

        fold_results.append(metrics)
        print(
            f"Fold {fold} student val: "
            f"macro_f1={metrics.get('macro_f1', float('nan')):.4f}, "
            f"accuracy={metrics.get('accuracy', float('nan')):.4f}"
        )

    summary: Dict[str, Any] = {"n_splits": n_splits, "folds": fold_results}
    print("\n===== K-FOLD RESULTS (student val) =====")

    for key in metric_keys:
        values = [
            float(result[key])
            for result in fold_results
            if key in result
        ]
        if not values:
            continue

        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        summary[f"{key}_mean"] = mean_value
        summary[f"{key}_std"] = std_value

        print(f"  {key}: {mean_value:.4f} ± {std_value:.4f}")
        wandb_log(
            {
                f"cv/{key}_mean": mean_value,
                f"cv/{key}_std": std_value,
            }
        )

    with (output_directory / "cv_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return summary


def train_pipeline(config: TrainingConfig) -> None:
    config.resolve_runtime()
    set_seed(config.seed)

    output_directory = Path(config.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Device: {config.device}")
    print(f"GPU ids: {config.gpu_ids}")
    print(f"Mixed precision: {config.use_mixed_precision}")

    train_df, valid_df, test_df, label2id, id2label = resolve_splits(config)

    print("Label mapping:")
    for label, label_id in label2id.items():
        count = int((train_df["verdict_label"] == label).sum())
        print(f"  {label_id}: {label}, số mẫu train={count}")

    test_size = 0 if test_df is None else len(test_df)
    print(
        f"Train={len(train_df)}, "
        f"Validation={len(valid_df)}, "
        f"Test={test_size}"
    )

    train_df.to_json(
        output_directory / "train_data.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    valid_df.to_json(
        output_directory / "valid_data.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    if test_df is not None:
        test_df.to_json(
            output_directory / "test_data.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )

    configuration = save_training_configuration(
        output_directory=output_directory,
        config=config,
        label2id=label2id,
        id2label=id2label,
    )

    initialize_wandb(config, configuration)
    reset_global_step()

    try:
        teacher_tokenizer = load_tokenizer(
            config.teacher_model_name,
            config.trust_remote_code,
        )
        student_tokenizer = load_tokenizer(
            config.student_model_name,
            config.trust_remote_code,
        )

        if config.kfold and config.kfold > 1:
            frames = [train_df, valid_df]
            if test_df is not None:
                frames.append(test_df)
            combined_df = pd.concat(frames, ignore_index=True)

            run_cross_validation(
                config=config,
                dataframe=combined_df,
                label2id=label2id,
                id2label=id2label,
                output_directory=output_directory,
                teacher_tokenizer=teacher_tokenizer,
                student_tokenizer=student_tokenizer,
            )

            print(
                "\nHuấn luyện model cuối cùng trên split chuẩn "
                "để lấy artifact triển khai..."
            )

        train_dataset = VerdictDataset(train_df, label2id)
        valid_dataset = VerdictDataset(valid_df, label2id)

        collator = DistillationCollator(
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            teacher_max_length=config.teacher_max_length,
            student_max_length=config.student_max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = None
        if test_df is not None:
            test_dataset = VerdictDataset(test_df, label2id)
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collator,
                pin_memory=torch.cuda.is_available(),
            )

        class_weights = calculate_class_weights(
            train_df, label2id, config.device
        )

        print("Class weights:", class_weights.detach().cpu().tolist())

        num_labels = len(label2id)

        print("\nKhởi tạo teacher...")
        teacher = load_classification_model(
            model_name=config.teacher_model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            config=config,
        )

        teacher_output = train_teacher(
            teacher=teacher,
            train_loader=train_loader,
            valid_loader=valid_loader,
            class_weights=class_weights,
            id2label=id2label,
            output_directory=output_directory,
            config=config,
        )

        del teacher

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Teacher tốt nhất được lưu tại: {teacher_output}")

        teacher = load_pretrained_model(teacher_output, config)
        teacher.eval()

        print("\nKhởi tạo student...")
        student = load_classification_model(
            model_name=config.student_model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            config=config,
        )

        student_output = train_student(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            valid_loader=valid_loader,
            class_weights=class_weights,
            id2label=id2label,
            student_tokenizer=student_tokenizer,
            output_directory=output_directory,
            config=config,
        )

        del student

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Student tốt nhất được lưu tại: {student_output}")

        if test_loader is not None:
            teacher_test_metrics, teacher_test_labels, (
                teacher_test_predictions
            ) = evaluate_model(
                teacher,
                test_loader,
                input_type="teacher",
                id2label=id2label,
                device=config.device,
            )

            save_evaluation_results(
                metrics=teacher_test_metrics,
                labels=teacher_test_labels,
                predictions=teacher_test_predictions,
                id2label=id2label,
                output_directory=teacher_output,
                prefix="test",
            )

            print(
                "\nTeacher test:"
                f"\n  Accuracy: {teacher_test_metrics['accuracy']:.4f}"
                f"\n  Macro precision: "
                f"{teacher_test_metrics['macro_precision']:.4f}"
                f"\n  Macro recall: "
                f"{teacher_test_metrics['macro_recall']:.4f}"
                f"\n  Macro F1: {teacher_test_metrics['macro_f1']:.4f}"
            )

            wandb_log(
                {
                    "teacher/test_accuracy": teacher_test_metrics["accuracy"],
                    "teacher/test_macro_precision": teacher_test_metrics[
                        "macro_precision"
                    ],
                    "teacher/test_macro_recall": teacher_test_metrics[
                        "macro_recall"
                    ],
                    "teacher/test_macro_f1": teacher_test_metrics["macro_f1"],
                    "teacher/test_weighted_f1": teacher_test_metrics[
                        "weighted_f1"
                    ],
                }
            )

            best_student = load_pretrained_model(student_output, config)

            student_test_metrics, student_test_labels, (
                student_test_predictions
            ) = evaluate_model(
                best_student,
                test_loader,
                input_type="student",
                id2label=id2label,
                device=config.device,
            )

            save_evaluation_results(
                metrics=student_test_metrics,
                labels=student_test_labels,
                predictions=student_test_predictions,
                id2label=id2label,
                output_directory=student_output,
                prefix="test",
            )

            print(
                "\nStudent test:"
                f"\n  Accuracy: {student_test_metrics['accuracy']:.4f}"
                f"\n  Macro precision: "
                f"{student_test_metrics['macro_precision']:.4f}"
                f"\n  Macro recall: "
                f"{student_test_metrics['macro_recall']:.4f}"
                f"\n  Macro F1: {student_test_metrics['macro_f1']:.4f}"
                f"\n  Weighted F1: {student_test_metrics['weighted_f1']:.4f}"
                f"\n  Balanced accuracy: "
                f"{student_test_metrics['balanced_accuracy']:.4f}"
            )

            wandb_log(
                {
                    "student/test_accuracy": student_test_metrics["accuracy"],
                    "student/test_macro_precision": student_test_metrics[
                        "macro_precision"
                    ],
                    "student/test_macro_recall": student_test_metrics[
                        "macro_recall"
                    ],
                    "student/test_macro_f1": student_test_metrics["macro_f1"],
                    "student/test_weighted_f1": student_test_metrics[
                        "weighted_f1"
                    ],
                    "student/test_balanced_accuracy": student_test_metrics[
                        "balanced_accuracy"
                    ],
                }
            )
        else:
            print(
                "\nKhông có test set nội bộ. "
                "Dùng file predict để đánh giá trên test."
            )

        push_student_to_hub(config, student_output)

        print(
            "\nHoàn tất huấn luyện."
            f"\nStudent dùng inference: {student_output}"
        )

    finally:
        finish_wandb()


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
        description="Train teacher và distill student cho verdict classification."
    )

    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--valid-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./verdict_outputs")

    parser.add_argument(
        "--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL_NAME
    )
    parser.add_argument(
        "--student-model", type=str, default=DEFAULT_STUDENT_MODEL_NAME
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Danh sách GPU, ví dụ '0' hoặc '0,1'.",
    )

    parser.add_argument("--teacher-max-length", type=int, default=2048)
    parser.add_argument("--student-max-length", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--student-epochs", type=int, default=8)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    parser.add_argument("--teacher-lr", type=float, default=2e-5)
    parser.add_argument("--student-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["cosine", "linear"],
        default="cosine",
    )

    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--early-stopping-patience", type=int, default=3)

    parser.add_argument("--freeze-encoder-layers", type=int, default=0)
    parser.add_argument("--freeze-embeddings", action="store_true")

    parser.add_argument(
        "--kfold",
        type=int,
        default=0,
        help="Số fold cross-validation (>=2 để bật). 0 = tắt.",
    )

    parser.add_argument(
        "--split-mode", type=str, choices=["random", "time"], default="random"
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="alqac-legal-outcome-distillation",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=HUB_MODEL_ID,
    )
    parser.add_argument("--hub-public", action="store_true")
    parser.add_argument("--hub-license", type=str, default="apache-2.0")
    parser.add_argument("--no-push", action="store_true")

    return parser


def build_config_from_args(
    arguments: argparse.Namespace,
) -> TrainingConfig:
    teacher_epochs = arguments.teacher_epochs
    student_epochs = arguments.student_epochs

    if arguments.epochs is not None:
        teacher_epochs = arguments.epochs
        student_epochs = arguments.epochs

    return TrainingConfig(
        data_path=arguments.data,
        train_data_path=arguments.train_data,
        valid_data_path=arguments.valid_data,
        test_data_path=arguments.test_data,
        output_dir=arguments.output_dir,
        teacher_model_name=arguments.teacher_model,
        student_model_name=arguments.student_model,
        trust_remote_code=arguments.trust_remote_code,
        seed=arguments.seed,
        teacher_max_length=arguments.teacher_max_length,
        student_max_length=arguments.student_max_length,
        teacher_epochs=teacher_epochs,
        student_epochs=student_epochs,
        train_batch_size=arguments.batch_size,
        eval_batch_size=arguments.eval_batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        teacher_learning_rate=arguments.teacher_lr,
        student_learning_rate=arguments.student_lr,
        weight_decay=arguments.weight_decay,
        warmup_ratio=arguments.warmup_ratio,
        max_grad_norm=arguments.max_grad_norm,
        label_smoothing=arguments.label_smoothing,
        lr_scheduler_type=arguments.lr_scheduler,
        distill_ce_weight=arguments.ce_weight,
        distill_alpha=arguments.alpha,
        distill_beta=arguments.beta,
        distill_temperature=arguments.temperature,
        early_stopping_patience=arguments.early_stopping_patience,
        freeze_encoder_layers=arguments.freeze_encoder_layers,
        freeze_embeddings=arguments.freeze_embeddings,
        kfold=arguments.kfold,
        split_mode=arguments.split_mode,
        train_ratio=arguments.train_ratio,
        valid_ratio=arguments.valid_ratio,
        test_ratio=arguments.test_ratio,
        num_workers=arguments.num_workers,
        gpu_ids=parse_gpu_ids(arguments.gpu_ids),
        use_wandb=not arguments.no_wandb,
        wandb_project=arguments.wandb_project_name,
        wandb_entity=arguments.wandb_entity,
        wandb_run_name=arguments.wandb_run_name,
        push_to_hub=not arguments.no_push,
        hub_model_id=arguments.hub_model_id,
        hub_private=not arguments.hub_public,
        hub_license=arguments.hub_license,
    )


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    has_explicit_split = bool(arguments.train_data and arguments.valid_data)

    if not arguments.data and not has_explicit_split:
        parser.error(
            "Cần --data để tự chia, hoặc cả --train-data và --valid-data."
        )

    config = build_config_from_args(arguments)

    train_pipeline(config)


if __name__ == "__main__":
    main()
