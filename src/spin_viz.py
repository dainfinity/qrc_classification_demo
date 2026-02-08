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


def _build_edges(n_spins: int, topology: str) -> List[Tuple[int, int]]:
    if topology == "all_to_all":
        return [(i, j) for i in range(n_spins) for j in range(i + 1, n_spins)]
    if topology == "chain_1d":
        return [(i, i + 1) for i in range(n_spins - 1)]
    raise ValueError(f"Unknown topology: {topology}")


def _build_positions(n_spins: int, topology: str) -> np.ndarray:
    if topology == "chain_1d":
        xs = np.linspace(0.0, 1.0, n_spins)
        ys = np.zeros(n_spins)
        return np.stack([xs, ys], axis=1)
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
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="#0b0f19")
    # Fixed range: changes in J width stay visually comparable.
    vis_j_max = 2.0
    cmap_edges = plt.cm.RdYlBu_r
    cmap_nodes = plt.cm.cividis

    x = snapshot.positions[:, 0]
    y = snapshot.positions[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xpad = 0.25 * max(1.0, xmax - xmin)
    ypad = 0.25 * max(1.0, ymax - ymin)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    # Subtle gradient backdrop to make couplings stand out.
    grad_x = np.linspace(0.0, 1.0, 240)
    grad = np.outer(np.ones(140), grad_x)
    ax.imshow(
        grad,
        extent=[xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad],
        origin="lower",
        cmap="magma",
        alpha=0.13,
        zorder=0,
        aspect="auto",
    )

    for (i, j), value in snapshot.couplings.items():
        x1, y1 = snapshot.positions[i]
        x2, y2 = snapshot.positions[j]
        norm_abs = np.clip(abs(value) / vis_j_max, 0.0, 1.0)
        norm_signed = np.clip(value / vis_j_max, -1.0, 1.0)
        strength = 0.7 + 3.2 * norm_abs
        color = cmap_edges(0.5 + 0.5 * norm_signed)
        # Glow stroke + sharp stroke for a cleaner visual.
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=strength + 2.0, alpha=0.18, zorder=1)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=strength, alpha=0.95, zorder=2)

    node_metric = np.clip(snapshot.h_field / 2.0, 0.0, 1.0)
    node_colors = [cmap_nodes(node_metric) for _ in range(snapshot.n_spins)]
    # Soft halo
    ax.scatter(
        snapshot.positions[:, 0],
        snapshot.positions[:, 1],
        s=610,
        c=node_colors,
        edgecolors="none",
        alpha=0.22,
        zorder=3,
    )
    ax.scatter(
        snapshot.positions[:, 0],
        snapshot.positions[:, 1],
        s=280,
        c=node_colors,
        edgecolors="#f7f7f7",
        linewidths=1.0,
        zorder=4,
    )
    for idx, (x, y) in enumerate(snapshot.positions):
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=9, color="#111111", zorder=5)

    ax.set_title(
        f"N={snapshot.n_spins}, topology={snapshot.topology}, J~U[-{snapshot.j_width:.2f}, {snapshot.j_width:.2f}], h={snapshot.h_field:.2f}",
        fontsize=11,
        color="#f5f5f5",
        pad=10,
    )
    ax.text(
        0.02,
        0.02,
        "edge color: coupling sign/strength   |   node color: transverse field h",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#dddddd",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    plt.show()


def build_spin_control_panel(default_n: int = 8, seed: int = 1234) -> tuple[VBox, dict]:
    n_slider = IntSlider(value=int(default_n), min=4, max=20, step=1, description="N")
    j_slider = FloatSlider(value=0.8, min=0.0, max=2.0, step=0.02, description="J width")
    h_slider = FloatSlider(value=0.8, min=0.0, max=2.0, step=0.02, description="h field")
    topo_dropdown = Dropdown(
        options=[("All-to-all", "all_to_all"), ("1D Chain", "chain_1d")],
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
