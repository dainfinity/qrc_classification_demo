"""Create fixed train/test CSV files for a binary time-series classification task.

This script intentionally uses only the Python standard library so it can run
even before a virtual environment is prepared.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path


def _make_class_samples(
    rng: random.Random,
    m: int,
    length: int,
    freq_range: tuple[float, float],
    noise_std: float,
) -> list[list[float]]:
    t = [k / (length - 1) for k in range(length)]
    samples = []
    for _ in range(m):
        freq = rng.uniform(freq_range[0], freq_range[1])
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amp = rng.uniform(0.7, 1.3)
        seq = []
        for tk in t:
            value = amp * math.sin(2.0 * math.pi * freq * tk + phase)
            value += rng.gauss(0.0, noise_std)
            seq.append(value)
        samples.append(seq)
    return samples


def _shuffle_xy(
    rng: random.Random, x: list[list[float]], y: list[int]
) -> tuple[list[list[float]], list[int]]:
    paired = list(zip(x, y))
    rng.shuffle(paired)
    x_shuf, y_shuf = zip(*paired)
    return [list(row) for row in x_shuf], list(y_shuf)


def _write_matrix_csv(path: Path, rows: list[list[float]], prefix: str = "t") -> None:
    header = [f"{prefix}{k}" for k in range(len(rows[0]))]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_label_csv(path: Path, labels: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"])
        for y in labels:
            writer.writerow([y])


def make_dataset(
    out_dir: Path,
    m_train: int = 100,
    m_test: int = 100,
    length: int = 96,
    seed: int = 2026,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    train_half = m_train // 2
    test_half = m_test // 2

    x0_train = _make_class_samples(rng, train_half, length, (1.0, 2.0), 0.10)
    x1_train = _make_class_samples(rng, m_train - train_half, length, (3.0, 4.2), 0.10)
    x0_test = _make_class_samples(rng, test_half, length, (1.0, 2.0), 0.10)
    x1_test = _make_class_samples(rng, m_test - test_half, length, (3.0, 4.2), 0.10)

    x_train = x0_train + x1_train
    y_train = [0] * len(x0_train) + [1] * len(x1_train)
    x_test = x0_test + x1_test
    y_test = [0] * len(x0_test) + [1] * len(x1_test)

    x_train, y_train = _shuffle_xy(rng, x_train, y_train)
    x_test, y_test = _shuffle_xy(rng, x_test, y_test)

    _write_matrix_csv(out_dir / "train_X.csv", x_train, prefix="t")
    _write_label_csv(out_dir / "train_y.csv", y_train)
    _write_matrix_csv(out_dir / "test_X.csv", x_test, prefix="t")
    _write_label_csv(out_dir / "test_y.csv", y_test)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    make_dataset(out_dir=root / "data")
