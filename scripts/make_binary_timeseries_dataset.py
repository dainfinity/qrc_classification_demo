"""Create fixed train/test CSV files for a binary time-series classification task.

This script intentionally uses only the Python standard library so it can run
even before a virtual environment is prepared.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path


def _make_order_swap_samples(
    rng: random.Random,
    m: int,
    length: int,
    label: int,
    noise_std: float,
    switch_range: tuple[float, float],
) -> list[list[float]]:
    """Generate samples with identical components but opposite temporal order.

    class 0: low-frequency (early) -> high-frequency (late)
    class 1: high-frequency (early) -> low-frequency (late)

    The swap time is randomized per sample within switch_range. Set a fixed
    range (e.g., 0.5 to 0.5) to keep the switch at a constant time.
    """
    t = [k / (length - 1) for k in range(length)]
    samples: list[list[float]] = []
    switch_min, switch_max = switch_range
    for _ in range(m):
        f_low = rng.uniform(1.7, 2.5)
        f_high = rng.uniform(2.4, 3.2)
        phase_low = rng.uniform(0.0, 2.0 * math.pi)
        phase_high = rng.uniform(0.0, 2.0 * math.pi)
        amp_low = rng.uniform(0.75, 1.15)
        amp_high = rng.uniform(0.75, 1.15)
        trend = rng.uniform(-0.15, 0.15)
        sharpness = rng.uniform(12.0, 20.0)
        switch_t = rng.uniform(switch_min, switch_max)

        seq = []
        for tk in t:
            gate = 1.0 / (1.0 + math.exp(-sharpness * (tk - switch_t)))
            if label == 0:
                w_low, w_high = 1.0 - gate, gate
            else:
                w_low, w_high = gate, 1.0 - gate

            low_component = amp_low * math.sin(2.0 * math.pi * f_low * tk + phase_low)
            high_component = amp_high * math.sin(2.0 * math.pi * f_high * tk + phase_high)
            value = w_low * low_component + w_high * high_component
            value += trend * (tk - 0.5)
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
    switch_range: tuple[float, float] = (0.5, 0.5),
    noise_std: float = 0.8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    train_half = m_train // 2
    test_half = m_test // 2

    x0_train = _make_order_swap_samples(
        rng, train_half, length, label=0, noise_std=noise_std, switch_range=switch_range
    )
    x1_train = _make_order_swap_samples(
        rng, m_train - train_half, length, label=1, noise_std=noise_std, switch_range=switch_range
    )
    x0_test = _make_order_swap_samples(
        rng, test_half, length, label=0, noise_std=noise_std, switch_range=switch_range
    )
    x1_test = _make_order_swap_samples(
        rng, m_test - test_half, length, label=1, noise_std=noise_std, switch_range=switch_range
    )

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
