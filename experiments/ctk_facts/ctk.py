import os
import json
from typing import List


import sqlite3
import pandas as pd
from datasets.dataset_dict import Dataset, DatasetDict
from rich.progress import Progress

from datasets import Dataset, DatasetDict, load_from_disk

"""
    Following methods were taken from https://github.com/aic-factcheck/long-input-gnn/tree/main repository
"""

def load_jsonl(filepath: str, encoding="utf-8", ensure_ascii=False):
    output = []

    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            record = json.loads(line)
            output.append(record)

    return output


def get_evidence(x, ARTICLES_DB_PATH, k=None):
    k = min(k, len(x["predicted_pages"])) if k else len(x["predicted_pages"])
    evidence_ids = x["predicted_pages"][:k]

    with sqlite3.connect(ARTICLES_DB_PATH) as con:
        df = pd.read_sql_query("SELECT * FROM documents WHERE id IN ({})".format(
            "'" + "', '".join(evidence_ids) + "'"
        ), con).set_index("id")

        return df.loc[evidence_ids]["text"].to_list()


def convert_list_to_dict(dataset: list, ignore_fields: list = None):
    keys = [ key for key in dataset[0].keys() if key not in ignore_fields ]
    output = { key: [] for key in keys }
    for x in dataset:
        for key in keys:
            output[key].append(x[key])

    return output


def load_ctknews(paths: dict, k: int = 10):
    LABELS = { "REFUTES": 0, "SUPPORTS": 1, "NOT ENOUGH INFO": 2 }

    splits = {
        name: load_jsonl(path)
        for name, path in paths.items()
    }

    for split_name in splits.keys():
        for example_idx in range(len(splits[split_name])):
            splits[split_name][example_idx]["label"] = LABELS[
                splits[split_name][example_idx]["label"]
            ]
            splits[split_name][example_idx]["evidence"] = get_evidence(
                splits[split_name][example_idx], k=k
            )

    splits = {
        k: convert_list_to_dict(v, ignore_fields=["id", "verifiable", "predicted_pages"])
        for k, v in splits.items()
    }

    splits = {
        k: Dataset.from_dict(v)
        for k, v in splits.items()
    }

    return DatasetDict(splits)