"""
시작/목표 + 적 배치 시나리오 여러 개 → 각각 A*(base cost) 경로 + 채점 + 시각화.
  python scenarios.py

각 시나리오마다 fin/runs/<ts>_scn_<label>/ 폴더 1개 (full_run 기록 일체).
경로는 지형 base cost A* (적 비반응). 적 위협장에 얹어 생존 점수 비교.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
from tanknav import mapio, risk, planning, eval as ev, runlog


def P(r0, c0, r1, c1):
    """patrol 경로 셀 생성 헬퍼"""
    return risk.bresenham(r0, c0, r1, c1)


# (label, start, goal, enemies)
def scenarios():
    return [
        ("diagNWSE", (5, 5), (55, 55), [
            risk.Enemy("tank",     [(30, 30)]),
            risk.Enemy("patrol",   P(45, 10, 45, 50))]),
        ("cornerSWNE", (55, 5), (5, 55), [
            risk.Enemy("tank",     [(25, 25)]),
            risk.Enemy("infantry", [(40, 40)]),
            risk.Enemy("patrol",   P(10, 20, 10, 50))]),
        ("gauntletNS", (5, 30), (55, 30), [
            risk.Enemy("infantry", [(30, 15)]),
            risk.Enemy("infantry", [(30, 45)]),
            risk.Enemy("patrol",   P(40, 20, 40, 40))]),
        ("horizWE", (30, 5), (30, 55), [
            risk.Enemy("tank",     [(15, 30)]),
            risk.Enemy("patrol",   P(45, 10, 45, 45))]),
        ("mostlysafe", (5, 5), (55, 55), [
            risk.Enemy("tank",     [(52, 52)]),
            risk.Enemy("patrol",   P(50, 45, 58, 45))]),
    ]


def main():
    bundle = mapio.load_maps()
    hm, cost = bundle.heightmap_filled, bundle.cost
    print(f"[map] {bundle.ts}  {cost.shape}\n")

    summary = []
    for label, start, goal, enemies in scenarios():
        if not (np.isfinite(cost[start]) and np.isfinite(cost[goal])):
            print(f"  ⚠️  {label}: start/goal 통과불가 — 건너뜀")
            continue
        path = planning.astar(start, goal, cost, hm)
        if path is None:
            print(f"  ⚠️  {label}: 경로 없음 — 건너뜀")
            continue

        run_dir = runlog.new_run(f"scn_{label}")
        named = {f"astar_{label}": path}
        precomp, rows = runlog.full_run(run_dir, bundle, enemies, named)

        surv = next(r for r in rows if r["mission"] == "survival")
        summary.append((label, start, goal, len(path),
                        surv["P_survive"], surv["time_s"], run_dir.name))
        print(f"  [{label:11s}] {start}→{goal}  len {len(path):3d}  "
              f"P_surv {surv['P_survive']:.3f}  time {surv['time_s']:.0f}s  → {run_dir.name}")

    print(f"\n{'scenario':12s} {'start':8s} {'goal':8s} {'len':>4s} {'P_surv':>7s} {'time_s':>7s}")
    for label, s, g, n, ps, t, _ in summary:
        print(f"{label:12s} {str(s):8s} {str(g):8s} {n:4d} {ps:7.3f} {t:7.0f}")


if __name__ == "__main__":
    main()
