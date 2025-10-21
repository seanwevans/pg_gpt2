"""Utility script to build visual assets describing how GPT-2 lives inside PostgreSQL."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class TrainingSnapshot:
    epoch: int
    loss: float
    grad_norm: float
    clip_threshold: float


def synthetic_training_history(num_epochs: int = 20) -> List[TrainingSnapshot]:
    """Return a smooth synthetic training curve for demonstration purposes."""
    epochs = np.arange(1, num_epochs + 1)
    # Simulate loss decay with some noise.
    loss = 3.5 * np.exp(-epochs / 8) + 0.1 * np.sin(epochs / 1.5)
    # Simulate gradient norms that oscillate around a clipping threshold.
    grad_norm = 1.4 + 0.3 * np.sin(epochs / 2.5) + 0.05 * np.random.default_rng(42).normal(size=num_epochs)
    clip_threshold = np.full_like(grad_norm, 1.5)

    return [
        TrainingSnapshot(int(epoch), float(l), float(g), float(c))
        for epoch, l, g, c in zip(epochs, loss, grad_norm, clip_threshold)
    ]


def render_training_metrics(output_dir: Path) -> Path:
    """Render a PNG chart summarizing training progress."""
    history = synthetic_training_history()
    epochs = [s.epoch for s in history]
    loss = [s.loss for s in history]
    grad_norm = [s.grad_norm for s in history]
    clip_threshold = [s.clip_threshold for s in history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.set_title("PG GPT-2 Training Progress")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, loss, marker="o", color="tab:blue", label="Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Gradient Norm", color="tab:orange")
    ax2.plot(epochs, grad_norm, marker="s", color="tab:orange", label="Gradient Norm")
    ax2.plot(epochs, clip_threshold, linestyle="--", color="tab:red", label="Clip Threshold")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    output_path = output_dir / "training_metrics.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


@dataclass
class FlowStage:
    table: str
    description: str


def gradient_flow_stages() -> Iterable[FlowStage]:
    return [
        FlowStage("llm_param", "Lookup parameters for current tokens"),
        FlowStage("llm_tape", "Record activations and micro-batch state"),
        FlowStage("llm_clip", "Apply gradient clipping prior to storage"),
        FlowStage("llm_tape", "Accumulate gradient signals"),
        FlowStage("llm_param", "Write updated weights back to table"),
    ]


def render_gradient_flow_animation(output_dir: Path) -> Path:
    """Render an animated GIF illustrating gradient propagation through tables."""
    stages = list(gradient_flow_stages())
    tables = ["llm_param", "llm_tape", "llm_clip"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.axis("off")

    table_positions = {
        "llm_param": (1, 2.5),
        "llm_tape": (3, 1.2),
        "llm_clip": (5, 2.5),
    }
    table_boxes = {}

    def init_diagram():
        ax.clear()
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.axis("off")
        for name, (x, y) in table_positions.items():
            rect = plt.Rectangle((x - 0.6, y - 0.4), 1.2, 0.8, linewidth=2, edgecolor="gray", facecolor="white")
            ax.add_patch(rect)
            ax.text(x, y, name, ha="center", va="center", fontsize=12, fontweight="bold")
            table_boxes[name] = rect
        # Draw directional arrows representing data flow.
        ax.annotate("", xy=(2.4, 2.3), xytext=(1.6, 2.3), arrowprops=dict(arrowstyle="->", linewidth=2, color="tab:blue"))
        ax.annotate("", xy=(4.4, 2.3), xytext=(3.6, 2.3), arrowprops=dict(arrowstyle="->", linewidth=2, color="tab:blue"))
        ax.annotate("", xy=(3.0, 1.6), xytext=(3.0, 2.1), arrowprops=dict(arrowstyle="->", linewidth=2, color="tab:orange"))
        ax.annotate("", xy=(2.4, 1.1), xytext=(1.6, 1.1), arrowprops=dict(arrowstyle="->", linewidth=2, color="tab:green"))
        stage_text = ax.text(3, 3.5, "", ha="center", va="center", fontsize=12)
        return stage_text

    stage_text = init_diagram()

    def update(frame_index: int):
        stage = stages[frame_index]
        # Reset colors
        for table, rect in table_boxes.items():
            rect.set_facecolor("white")
            rect.set_edgecolor("gray")
        active_rect = table_boxes[stage.table]
        active_rect.set_facecolor("#cfe2f3")
        active_rect.set_edgecolor("#1f77b4")
        stage_text.set_text(f"Step {frame_index + 1}: {stage.description}")
        return (*table_boxes.values(), stage_text)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(stages),
        init_func=lambda: init_diagram(),
        blit=False,
        repeat=True,
        interval=1200,
    )

    output_path = output_dir / "gradient_flow.gif"
    anim.save(output_path, writer=PillowWriter(fps=1))
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/visualizations"),
        help="Directory where generated assets will be written.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = render_training_metrics(output_dir)
    animation_path = render_gradient_flow_animation(output_dir)

    print(f"Generated training metrics chart at {metrics_path}")
    print(f"Generated gradient flow animation at {animation_path}")


if __name__ == "__main__":
    main()
