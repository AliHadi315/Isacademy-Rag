"""Generate the demo corpus: two PDFs (text + figure + table) and a CSV.

Run:  python demo/make_demo_data.py

The PDFs are real PDFs produced by reportlab with a real matplotlib chart
embedded, so the extraction, image and table paths are all genuinely exercised
- nothing about the demo is stubbed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "demo"


def _chart(path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, 11))
    baseline = [71.2, 73.0, 74.6, 75.9, 76.8, 77.4, 77.9, 78.2, 78.4, 78.5]
    tuned = [71.2, 76.4, 80.9, 83.7, 85.6, 86.9, 87.7, 88.1, 88.3, 88.4]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(epochs, baseline, marker="o", linewidth=2, label="Baseline")
    ax.plot(epochs, tuned, marker="s", linewidth=2, label="Fine-tuned")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(65, 95)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def _research_pdf(out: Path, chart: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="Adaptive Fine-Tuning Study",
                            author="Isacademy Research Group")
    story = [
        Paragraph("Adaptive Fine-Tuning of Compact Language Models", styles["Title"]),
        Paragraph("Isacademy Research Group, 2026", styles["Italic"]),
        Spacer(1, 0.6 * cm),
        Paragraph("Abstract", styles["Heading2"]),
        Paragraph(
            "We evaluate adaptive fine-tuning on three compact transformer models across "
            "four downstream classification tasks. Validation accuracy improved from 71.2 "
            "percent to 88.4 percent after ten epochs of adaptive fine-tuning, an absolute "
            "gain of 17.2 points over the frozen baseline. Inference latency fell by 34 "
            "percent because the adapted models converged at a smaller effective width. "
            "No degradation was observed on the held-out robustness suite.",
            styles["BodyText"],
        ),
        Paragraph("1. Introduction", styles["Heading2"]),
        Paragraph(
            "Compact models are attractive for deployment but historically trail larger "
            "architectures on specialised tasks. Prior work attributes this gap to limited "
            "representational capacity. We test the competing hypothesis that the gap is "
            "largely an optimisation artefact and can be closed by adapting the learning "
            "schedule to per-layer gradient statistics.",
            styles["BodyText"],
        ),
        Paragraph("2. Method", styles["Heading2"]),
        Paragraph(
            "Each model was trained for ten epochs with a per-layer adaptive schedule. The "
            "baseline used a fixed learning rate of 3e-5. All runs used identical data "
            "splits and three random seeds; reported numbers are seed means. Evaluation "
            "used a held-out validation split of 4,200 examples.",
            styles["BodyText"],
        ),
        PageBreak(),
        Paragraph("3. Results", styles["Heading2"]),
        Paragraph(
            "Figure 1 shows validation accuracy per epoch for the baseline and the "
            "fine-tuned configuration. The adaptive schedule separates from the baseline "
            "by epoch three and the gap widens until epoch eight, after which both curves "
            "flatten.",
            styles["BodyText"],
        ),
        Spacer(1, 0.3 * cm),
        Image(str(chart), width=15 * cm, height=8.4 * cm),
        Paragraph(
            "Figure 1: Validation accuracy per training epoch, baseline versus adaptive "
            "fine-tuning.",
            styles["Italic"],
        ),
        Spacer(1, 0.5 * cm),
        Paragraph(
            "Table 1 reports the final metrics per model. The largest relative improvement "
            "was recorded on Model C, which gained 19.1 accuracy points.",
            styles["BodyText"],
        ),
        Spacer(1, 0.3 * cm),
    ]

    rows = [
        ["Model", "Params (M)", "Baseline Acc", "Tuned Acc", "Latency ms", "Gain"],
        ["Model A", "82", "71.2", "88.4", "41.3", "17.2"],
        ["Model B", "124", "74.8", "89.6", "58.7", "14.8"],
        ["Model C", "66", "68.3", "87.4", "33.1", "19.1"],
        ["Model D", "180", "76.1", "90.2", "77.4", "14.1"],
    ]
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story += [
        table,
        Paragraph("Table 1: Final metrics per model.", styles["Italic"]),
        PageBreak(),
        Paragraph("4. Discussion", styles["Heading2"]),
        Paragraph(
            "The results support the optimisation hypothesis: capacity was not the binding "
            "constraint at this scale. The 34 percent latency reduction is a secondary "
            "effect of earlier convergence and should not be read as an architectural claim.",
            styles["BodyText"],
        ),
        Paragraph("5. Limitations", styles["Heading2"]),
        Paragraph(
            "All four models share a single pre-training corpus, so the findings may not "
            "transfer to differently pre-trained families. The robustness suite covers only "
            "English. We did not measure energy consumption.",
            styles["BodyText"],
        ),
        Paragraph("6. Conclusion", styles["Heading2"]),
        Paragraph(
            "Adaptive per-layer fine-tuning closed most of the accuracy gap between compact "
            "and large models on these tasks, at lower inference cost. We recommend it as "
            "the default schedule for models under 200 million parameters.",
            styles["BodyText"],
        ),
    ]
    doc.build(story)
    return out


def _market_pdf(out: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="Deployment Cost Review",
                            author="Isacademy Operations")
    doc.build([
        Paragraph("Compact Model Deployment: Cost Review", styles["Title"]),
        Paragraph("Isacademy Operations, 2026", styles["Italic"]),
        Spacer(1, 0.6 * cm),
        Paragraph("Summary", styles["Heading2"]),
        Paragraph(
            "Serving costs fell 28 percent quarter over quarter after the compact models "
            "replaced the previous 7-billion-parameter endpoint. Average request latency "
            "fell from 240 ms to 158 ms. Two incidents were recorded, both caused by "
            "cold-start behaviour rather than by model quality.",
            styles["BodyText"],
        ),
        Paragraph("Contradiction with the research report", styles["Heading2"]),
        Paragraph(
            "Operations measured a 28 percent latency reduction in production, whereas the "
            "research report records 34 percent on the validation harness. The gap is "
            "attributed to network overhead that the offline harness does not include.",
            styles["BodyText"],
        ),
        Paragraph("Risks", styles["Heading2"]),
        Paragraph(
            "Cold starts remain the dominant tail-latency risk. Capacity headroom is "
            "currently 40 percent and is expected to fall below 20 percent next quarter.",
            styles["BodyText"],
        ),
    ])
    return out


def _dataset(out: Path) -> Path:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 120
    params = rng.normal(110, 38, n).clip(30, 220)
    depth = (params / 9 + rng.normal(0, 1.6, n)).clip(4, 30)
    train_hours = params * 0.19 + rng.normal(0, 2.4, n)
    accuracy = 62 + 0.11 * params + 0.7 * depth + rng.normal(0, 2.3, n)
    latency = 8 + 0.36 * params + rng.normal(0, 4.5, n)
    memory = 0.9 * params + rng.normal(0, 9, n)

    df = pd.DataFrame({
        "model_family": rng.choice(["alpha", "beta", "gamma"], n),
        "params_millions": params.round(1),
        "layers": depth.round(0).astype(int),
        "train_hours": train_hours.round(2),
        "accuracy": accuracy.round(2),
        "latency_ms": latency.round(2),
        "memory_mb": memory.round(1),
    })
    # a few genuine gaps, so the missing-value path is exercised
    df.loc[rng.choice(n, 6, replace=False), "train_hours"] = None
    df.loc[rng.choice(n, 3, replace=False), "memory_mb"] = None
    df.to_csv(out, index=False)
    return out


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    chart = _chart(DEMO / "_figure1.png")
    research = _research_pdf(DEMO / "research_report.pdf", chart)
    market = _market_pdf(DEMO / "deployment_cost_review.pdf")
    dataset = _dataset(DEMO / "model_benchmarks.csv")
    for path in (research, market, dataset):
        print("created", path.relative_to(ROOT), "-", path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
