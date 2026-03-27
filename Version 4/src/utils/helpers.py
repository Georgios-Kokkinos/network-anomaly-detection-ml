"""
helpers.py

Utility helpers for the Version 2 pipeline:
- project path helpers
- reproducibility (seeds)
- simple file save helpers
- text sanitization for labels/features
"""

from __future__ import annotations

import json
import random
import re
import string
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


UNSW_CLASS_NAME_MAP = {
    0: "Benign",
    1: "Analysis",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Fuzzers",
    6: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms",
}


def project_root() -> Path:
    """Return the Version 2 project root directory.

    Parameters:
    None

    Returns:
    Path: Project root path.
    """
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    """Create a directory if missing and return it.

    Parameters:
    path (Path): Directory path to create.

    Returns:
    Path: The same path, ensured to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow.

    Parameters:
    seed (int): Random seed.

    Returns:
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(obj: Any, path: Path) -> None:
    """Save a Python object as JSON (UTF-8).

    Parameters:
    obj (Any): Object to serialize.
    path (Path): Output JSON path.

    Returns:
    None
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as CSV without an index column.

    Parameters:
    df (pd.DataFrame): Dataframe to save.
    path (Path): Output CSV path.

    Returns:
    None
    """
    df.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    """Save a UTF-8 text file.

    Parameters:
    text (str): Text content.
    path (Path): Output text path.

    Returns:
    None
    """
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def sanitize_text(value: str) -> str:
    """
    Normalize text to remove non-ASCII or non-printable characters and
    collapse whitespace so class/feature names render cleanly in plots and CSVs.

    Parameters:
    value (str): Input text value.

    Returns:
    str: Cleaned, ASCII-safe text.
    """
    # Convert to text and normalize unicode (drops accents/specials).
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    # Keep ASCII printable characters only.
    text = text.encode("ascii", "ignore").decode("ascii")
    text = "".join(ch for ch in text if ch in string.printable)
    # Collapse whitespace into single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_int_label(value: Any) -> int | None:
    """Convert a label value to int when possible, else return None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        as_float = float(text)
    except ValueError:
        return None
    if as_float.is_integer():
        return int(as_float)
    return None


def map_unsw_class_names(raw_labels: list[Any]) -> list[str]:
    """Map UNSW numeric labels to canonical class names.

    Non-numeric labels are kept as cleaned strings.
    """
    mapped: list[str] = []
    for label in raw_labels:
        label_int = _to_int_label(label)
        if label_int in UNSW_CLASS_NAME_MAP:
            mapped.append(UNSW_CLASS_NAME_MAP[label_int])
        else:
            mapped.append(sanitize_text(str(label)).replace("_", " ").strip())
    return mapped
