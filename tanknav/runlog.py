"""
[Run 로거]  채점/튜닝 한 번 = 폴더 하나. 자기완결 기록.

fin/runs/<타임스탬프>[_label]/ 안에:
  params.json        — config 파라미터 스냅샷 (재현성)
  enemies.json       — 적 배치
  map_info.json      — 맵 ts/shape/위협 통계
  paths.json         — 채점한 경로들의 셀+미터 좌표
  scores.csv / .txt  — 경로×임무 점수
  per_segment.csv    — 경로 세그먼트별 위험도 분해 (intensity/conceal/cover/exposure/p_hit)
  threat_*.npy       — 위협층 (intensity/concealment/cover)
  threatmap.png      — 위협장 + 경로 시각화
결과가 fin/ 루트에 쌓이지 않게 격리.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

RUNS_DIR = config.DATA_DIR / "runs"

MISSIONS = ("survival", "attack", "defend", "recon")

_SNAP_KEYS = [
    "GRID_RES", "K2_HEIGHT_M",
    "SLOPE_TIER", "TIER_SPEED", "T_EXPOSE",
    "PK_BASE", "COVER_FLOOR",
    "DETECT_RANGE", "OBS_HEIGHT", "WEAPON_RANGE", "WEAPON_PH",
    "MISSION_WEIGHTS",
]


def new_run(label: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{label}" if label else ts
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 개별 저장기 ────────────────────────────────────────────
def snapshot_config(run_dir: Path) -> dict:
    snap = {k: getattr(config, k) for k in _SNAP_KEYS}
    (run_dir / "params.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return snap


def save_enemies(run_dir: Path, enemies) -> None:
    data = [{"etype": e.etype, "positions": [list(p) for p in e.positions]}
            for e in enemies]
    (run_dir / "enemies.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_map_info(run_dir: Path, bundle, precomp) -> None:
    it = precomp.intensity
    vis = it > 0
    info = {
        "map_ts": bundle.ts,
        "shape": list(it.shape),
        "grid_res_m": config.GRID_RES,
        "threat_cells_gt0": int(vis.sum()),
        "total_cells": int(it.size),
        "threat_coverage": round(float(vis.mean()), 4),
        "intensity_max": round(float(it.max()), 4),
        "intensity_mean_visible": round(float(it[vis].mean()) if vis.any() else 0.0, 4),
    }
    (run_dir / "map_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def save_paths(run_dir: Path, named_paths: dict) -> None:
    g = config.GRID_RES
    data = {name: {"cells": [[int(r), int(c)] for r, c in cells],
                   "meters": [[r * g, c * g] for r, c in cells]}
            for name, cells in named_paths.items()}
    (run_dir / "paths.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_scores(run_dir: Path, rows: list[dict]) -> None:
    cols = ["path", "mission", "P_survive", "time_s", "t_norm", "score"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    (run_dir / "scores.csv").write_text("\n".join(lines), encoding="utf-8")
    txt = [f"{'path':32s} {'mission':9s} {'P_surv':>7s} {'time_s':>8s} {'t_norm':>7s} {'score':>7s}"]
    last = None
    for r in rows:
        if last is not None and r["path"] != last:
            txt.append("")
        last = r["path"]
        txt.append(f"{r['path']:32s} {r['mission']:9s} {r['P_survive']:7.3f} "
                   f"{r['time_s']:8.1f} {r['t_norm']:7.2f} {r['score']:7.3f}")
    (run_dir / "scores.txt").write_text("\n".join(txt), encoding="utf-8")


def save_per_segment(run_dir: Path, rows: list[dict]) -> None:
    cols = ["path", "i", "row", "col", "tier", "dt",
            "intensity", "concealment", "cover", "exposure", "p_hit"]
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(",".join(vals))
    (run_dir / "per_segment.csv").write_text("\n".join(lines), encoding="utf-8")


def save_threat_layers(run_dir: Path, precomp) -> None:
    np.save(run_dir / "threat_intensity.npy", precomp.intensity)
    np.save(run_dir / "threat_concealment.npy", precomp.concealment)
    np.save(run_dir / "threat_cover.npy", precomp.cover)


def plot_threatmap(run_dir: Path, bundle, enemies, precomp, named_paths: dict) -> None:
    exposure1 = 1 - np.exp(-1.0 / config.T_EXPOSE)
    phit = (precomp.intensity * precomp.concealment * exposure1
            * config.PK_BASE * precomp.cover)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    im0 = ax[0].imshow(precomp.intensity, cmap="inferno", vmin=0, vmax=1, origin="upper")
    ax[0].set_title("Threat intensity (max over enemies)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(phit, cmap="inferno", origin="upper")
    ax[1].set_title("p_hit / sec")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    for a in ax:
        for e in enemies:
            rs = [p[0] for p in e.positions]; cs = [p[1] for p in e.positions]
            if e.etype == "patrol":
                a.plot(cs, rs, "c-", lw=2, alpha=.8)
            else:
                a.plot(cs, rs, "cs", ms=9)

    colors = ["lime", "deepskyblue", "magenta", "yellow", "white", "orange"]
    for (name, cells), col in zip(named_paths.items(), colors):
        rs = [c[0] for c in cells]; cs = [c[1] for c in cells]
        for a in ax:
            a.plot(cs, rs, color=col, lw=1.5, alpha=.7, label=name)
    ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "threatmap.png", dpi=110)
    plt.close(fig)


# ── 통합 오케스트레이터 ────────────────────────────────────
def full_run(run_dir: Path, bundle, enemies, named_paths: dict,
             missions=MISSIONS):
    """
    채점 핵심 기록 일체를 run_dir 에 저장.
    반환: (precomp, score_rows)  — 호출측이 추가 분석(민감도 등)에 재사용.
    """
    from . import eval as ev   # 순환참조 회피용 지연 import

    precomp = ev.precompute_threat(enemies, bundle)
    snapshot_config(run_dir)
    save_enemies(run_dir, enemies)
    save_map_info(run_dir, bundle, precomp)
    save_paths(run_dir, named_paths)
    save_threat_layers(run_dir, precomp)

    score_rows, seg_rows = [], []
    for name, cells in named_paths.items():
        base = ev.score_path(cells, enemies, bundle, "survival", precomp)
        for i, s in enumerate(base["per_seg"]):
            r, c = s["cell"]
            seg_rows.append({"path": name, "i": i, "row": r, "col": c,
                             "tier": s["tier"], "dt": s["dt"],
                             "intensity": s["intensity"], "concealment": s["concealment"],
                             "cover": s["cover"], "exposure": s["exposure"],
                             "p_hit": s["p_hit"]})
        for m in missions:
            res = ev.score_path(cells, enemies, bundle, m, precomp)
            score_rows.append({"path": name, "mission": m,
                               "P_survive": round(res["P_survive"], 4),
                               "time_s": round(res["time_s"], 1),
                               "t_norm": round(res["t_norm"], 3),
                               "score": round(res["score"], 4)})

    save_scores(run_dir, score_rows)
    save_per_segment(run_dir, seg_rows)
    plot_threatmap(run_dir, bundle, enemies, precomp, named_paths)
    return precomp, score_rows
