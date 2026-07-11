import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from legal_outcome_classification_train import (
    load_json_or_jsonl,
    random_split_dataframe,
)


def split_records(
    dataframe: pd.DataFrame,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return random_split_dataframe(
        dataframe,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )


def write_json(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = dataframe.to_dict(orient="records")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def print_distribution(name: str, dataframe: pd.DataFrame) -> None:
    counts = dataframe["verdict_label"].value_counts().to_dict()
    print(f"{name} ({len(dataframe)}): {counts}")


def run_split(
    data_path: str,
    output_dir: str,
    prefix: str,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    dataframe = load_json_or_jsonl(data_path)

    print(f"Tổng số bản ghi: {len(dataframe)}")

    train_df, valid_df, test_df = split_records(
        dataframe,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    output_directory = Path(output_dir)

    train_path = output_directory / f"{prefix}_train.json"
    valid_path = output_directory / f"{prefix}_val.json"
    test_path = output_directory / f"{prefix}_test.json"

    write_json(train_df, train_path)
    write_json(valid_df, valid_path)
    write_json(test_df, test_path)

    print_distribution("Train", train_df)
    print_distribution("Val", valid_df)
    print_distribution("Test", test_df)

    print(f"Đã lưu:\n  {train_path}\n  {valid_path}\n  {test_path}")

    return {
        "train": str(train_path),
        "val": str(valid_path),
        "test": str(test_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chia dữ liệu thành train/val/test cho outcome classification."
    )

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="outcome")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    run_split(
        data_path=arguments.data,
        output_dir=arguments.output_dir,
        prefix=arguments.prefix,
        seed=arguments.seed,
        train_ratio=arguments.train_ratio,
        valid_ratio=arguments.valid_ratio,
        test_ratio=arguments.test_ratio,
    )


if __name__ == "__main__":
    main()
