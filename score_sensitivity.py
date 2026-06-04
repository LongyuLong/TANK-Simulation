"""
생존 모델 민감도 — 보호 항(concealment/cover/exposure) ablation.
  python score_sensitivity.py [label]

full_run 핵심 기록(scores/threatmap/per_segment/...)에 더해
ablation 결과(sensitivity.txt/png)를 같은 폴더에 추가 저장.
mission=survival 기준(시간 무관).
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tanknav import mapio, risk, eval as ev, config, runlog

FACTORS = ["concealment", "cover", "exposure"]


def build_enemies():
    return [
        risk.Enemy("tank",     [(40, 18)]),
        risk.Enemy("infantry", [(20, 42)]),
        risk.Enemy("patrol",   risk.bresenham(48, 8, 48, 52)),
    ]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    run_dir = runlog.new_run(f"sens_{label}" if label else "sens")

    bundle = mapio.load_maps()
    enemies = build_enemies()

    path_files = sorted(glob.glob(str(config.DATA_DIR / "path_*.npy")))[-4:]
    named = {p.replace("\\", "/").split("/")[-1]: ev.load_path_cells(p)
             for p in path_files}

    # 핵심 기록 일체 (scores/threatmap/per_segment/threat_*.npy ...)
    precomp, _ = runlog.full_run(run_dir, bundle, enemies, named)

    # ── ablation 특화 분석 ──
    txt = []
    fig, axes = plt.subplots(1, len(named), figsize=(5 * len(named), 5), squeeze=False)

    for ax, (name, cells) in zip(axes[0], named.items()):
        base = ev.score_path(cells, enemies, bundle, "survival", precomp)["P_survive"]
        offs = {f: ev.score_path(cells, enemies, bundle, "survival", precomp,
                                 ablate=frozenset([f]))["P_survive"]
                for f in FACTORS}

        txt.append(f"=== {name} ===")
        txt.append(f"  P_survive(기준)        = {base:.3f}")
        for f in FACTORS:
            txt.append(f"  {f:12s} OFF → P={offs[f]:.3f}  (생존 기여 {base - offs[f]:+.3f})")
        txt.append("")

        labels = ["base"] + [f[:5] + "\noff" for f in FACTORS]
        vals = [base] + [offs[f] for f in FACTORS]
        ax.bar(labels, vals, color=["gray", "#d62728", "#1f77b4", "#2ca02c"])
        ax.set_ylim(0, 1)
        ax.set_title(name[-16:], fontsize=9)
        ax.set_ylabel("P_survive")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    report = "\n".join(txt)
    (run_dir / "sensitivity.txt").write_text(report, encoding="utf-8")
    fig.suptitle("생존 민감도 — 항 OFF 시 P_survive (낮을수록 그 항이 중요)", fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "sensitivity.png", dpi=110)
    plt.close(fig)

    print(f"[run] {run_dir}")
    print(report)


if __name__ == "__main__":
    main()
