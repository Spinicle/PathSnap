import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = REPO_ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()


def data_root() -> Path:
    root = os.environ.get("PATHSNAP_DATA", CFG["data"]["root"])
    p = Path(root)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def split_dir(split: str) -> Path:
    """split: 'train' | 'student_test' | 'eval'"""
    if split == "eval":
        env = os.environ.get("PATHSNAP_EVAL_DATA") or CFG["data"]["eval_dir"]
        if not env:
            raise RuntimeError("Set PATHSNAP_EVAL_DATA on the API laptop")
        return Path(env)
    key = {"train": "train_dir", "student_test": "test_dir"}[split]
    return data_root() / CFG["data"][key]
