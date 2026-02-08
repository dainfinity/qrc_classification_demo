"""Visualization helpers for interactive many-body spin systems in notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Dropdown, FloatSlider, HBox, IntSlider, VBox, interactive_output


@dataclass
class SpinSystemSnapshot:
    n_spins: int
    topology: str
    j_width: float
    h_field: float
    seed: int
    positions: np.ndarray
    edges: List[Tuple[int, int]]
    couplings: Dict[Tuple[int, int], float]


def _grid_shape(n_spins: int) -> Tuple[int, int]:
    rows = int(np.floor(np.sqrt(n_spins)))
    while rows > 1 and n_spins % rows != 0:
        rows -= 1
    cols = int(np.ceil(n_spins / rows))
    return rows, cols


def _build_edges(n_spins: int, topology: str) -> List[Tuple[int, int]]:
    if topology == "all_to_all":
        return [(i, j) for i in range(n_spins) for j in range(i + 1, n_spins)]
    if topology == "chain_1d":
        return [(i, i + 1) for i in range(n_spins - 1)]
    if topology == "grid_2d":
        rows, cols = _grid_shape(n_spins)
        edges = []
        for idx in range(n_spins):
            r, c = divmod(idx, cols)
            right = idx + 1
            down = idx + cols
            if c + 1 < cols and right < n_spins:
                edges.append((idx, right))
            if r + 1 < rows and down < n_spins:
                edges.append((idx, down))
        return edges
    raise ValueError(f"Unknown topology: {topology}")


def _build_positions(n_spins: int, topology: str) -> np.ndarray:
    if topology == "chain_1d":
        xs = np.linspace(0.0, 1.0, n_spins)
        ys = np.zeros(n_spins)
        return np.stack([xs, ys], axis=1)
    if topology == "grid_2d":
        rows, cols = _grid_shape(n_spins)
        pos = []
        for idx in range(n_spins):
            r, c = divmod(idx, cols)
            pos.append((c, -r))
        return np.array(pos, dtype=float)

    theta = np.linspace(0.0, 2.0 * np.pi, n_spins, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def sample_spin_system(
    n_spins: int,
    topology: str,
    j_width: float,
    h_field: float,
    seed: int = 1234,
) -> SpinSystemSnapshot:
    edges = _build_edges(n_spins=n_spins, topology=topology)
    positions = _build_positions(n_spins=n_spins, topology=topology)
    rng = np.random.default_rng(seed)
    couplings = {edge: float(rng.uniform(-j_width, j_width)) for edge in edges}
    return SpinSystemSnapshot(
        n_spins=n_spins,
        topology=topology,
        j_width=j_width,
        h_field=h_field,
        seed=seed,
        positions=positions,
        edges=edges,
        couplings=couplings,
    )


def draw_spin_snapshot(snapshot: SpinSystemSnapshot) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    max_abs_j = max(abs(v) for v in snapshot.couplings.values()) if snapshot.couplings else 1.0
    cmap_edges = plt.cm.coolwarm
    cmap_nodes = plt.cm.viridis

    for (i, j), value in snapshot.couplings.items():
        x1, y1 = snapshot.positions[i]
        x2, y2 = snapshot.positions[j]
        strength = 0.5 + 2.5 * abs(value) / (max_abs_j + 1e-12)
        color = cmap_edges(0.5 + 0.5 * value / (max_abs_j + 1e-12))
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=strength, alpha=0.9, zorder=1)

    node_metric = np.clip(np.abs(snapshot.h_field) / 3.0, 0.0, 1.0)
    node_colors = [cmap_nodes(node_metric) for _ in range(snapshot.n_spins)]
    ax.scatter(
        snapshot.positions[:, 0],
        snapshot.positions[:, 1],
        s=280,
        c=node_colors,
        edgecolors="black",
        linewidths=0.8,
        zorder=2,
    )
    for idx, (x, y) in enumerate(snapshot.positions):
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=9, color="white", zorder=3)

    ax.set_title(
        f"N={snapshot.n_spins}, topology={snapshot.topology}, J~U[-{snapshot.j_width:.2f}, {snapshot.j_width:.2f}], h={snapshot.h_field:.2f}",
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()


def build_spin_control_panel(default_n: int = 8, seed: int = 1234) -> tuple[VBox, dict]:
    n_slider = IntSlider(value=int(default_n), min=4, max=20, step=1, description="N")
    j_slider = FloatSlider(value=0.8, min=0.05, max=2.0, step=0.05, description="J width")
    h_slider = FloatSlider(value=0.8, min=0.0, max=3.0, step=0.05, description="h field")
    topo_dropdown = Dropdown(
        options=[("All-to-all", "all_to_all"), ("1D Chain", "chain_1d"), ("2D NN", "grid_2d")],
        value="all_to_all",
        description="Topology",
    )

    def _update(n_spins: int, j_width: float, h_field: float, topology: str) -> None:
        snapshot = sample_spin_system(
            n_spins=n_spins,
            topology=topology,
            j_width=float(j_width),
            h_field=float(h_field),
            seed=seed,
        )
        draw_spin_snapshot(snapshot)

    out = interactive_output(
        _update,
        {
            "n_spins": n_slider,
            "j_width": j_slider,
            "h_field": h_slider,
            "topology": topo_dropdown,
        },
    )
    ui = VBox([HBox([topo_dropdown, n_slider]), HBox([j_slider, h_slider])])
    panel = VBox([ui, out])
    controls = {
        "n_spins": n_slider,
        "j_width": j_slider,
        "h_field": h_slider,
        "topology": topo_dropdown,
        "seed": seed,
    }
    return panel, controls


def launch_spin_widget(default_n: int = 8, seed: int = 1234) -> VBox:
    panel, _ = build_spin_control_panel(default_n=default_n, seed=seed)
    return panel
