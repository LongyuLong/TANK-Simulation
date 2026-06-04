"""
[전역 계획] A* 글로벌 패스 플래너   (구 global_planner.py 통합)

이전 버전은 cost_map/heightmap/ROWS/COLS 를 모듈 전역으로 썼음.
여기서는 전부 인자로 받아 재사용 가능하게 파라미터화.

로컬 플래너(local.py, DWA 등)는 이 모듈이 만든 경로(waypoints)를
입력으로 받아 동작하게 확장 예정.
"""
from __future__ import annotations
import heapq
import numpy as np

from . import config, mapio, viz, mapping


def _heuristic(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _neighbors(r, c, rows, cols):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc, np.sqrt(dr * dr + dc * dc)


def astar(start, goal, cost_map, heightmap,
          height_penalty: float = config.HEIGHT_PENALTY):
    """
    A* 경로 탐색.
      start, goal : (row, col)
      cost_map    : 셀 비용 (inf=통과불가)
      heightmap   : 고도차 패널티용
    반환: path(list of (row,col)) 또는 None
    """
    rows, cols = cost_map.shape
    if np.isinf(cost_map[start]):
        raise ValueError(f"출발지 {start} 통과 불가")
    if np.isinf(cost_map[goal]):
        raise ValueError(f"목적지 {goal} 통과 불가")

    open_set = [(0.0, start)]
    came_from = {}
    g = np.full((rows, cols), np.inf)
    g[start] = 0.0

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]

        r, c = cur
        for nr, nc, step in _neighbors(r, c, rows, cols):
            cell = cost_map[nr, nc]
            if np.isinf(cell):
                continue
            dh = abs(float(heightmap[nr, nc]) - float(heightmap[r, c]))
            tentative = g[r, c] + step * cell + dh * height_penalty
            if tentative < g[nr, nc]:
                came_from[(nr, nc)] = cur
                g[nr, nc] = tentative
                f = tentative + _heuristic((nr, nc), goal)
                heapq.heappush(open_set, (f, (nr, nc)))
    return None


def has_line_of_sight(cost_map, a, b):
    """
    Theta* 스타일 Line-of-Sight 검사.

    a, b: (row, col)
    두 지점을 직선으로 연결했을 때 중간에 cost=inf 셀이 없으면 True.

    돌 본체, 절벽, 통행불가 obstacle은 cost=inf이므로
    이 함수에서 자동으로 차단된다.
    """
    r0, c0 = int(a[0]), int(a[1])
    r1, c1 = int(b[0]), int(b[1])

    rows, cols = cost_map.shape

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    err = dr - dc
    r, c = r0, c0

    while True:
        if not (0 <= r < rows and 0 <= c < cols):
            return False

        if not np.isfinite(cost_map[r, c]):
            return False

        if (r, c) == (r1, c1):
            return True

        e2 = 2 * err

        if e2 > -dc:
            err -= dc
            r += sr

        if e2 < dr:
            err += dr
            c += sc


def segment_cost(cost_map, a, b):
    """
    직선 a→b가 통과하는 셀들의 비용 적분.
      = (통과셀 평균비용) × (유클리드 거리[셀])
    중간에 inf(통행불가) 셀이 있으면 inf 반환(→ 단축 거부).
    원본 격자구간과 직선 단축을 동일 척도로 비교하기 위한 함수.
    """
    r0, c0 = int(a[0]), int(a[1])
    r1, c1 = int(b[0]), int(b[1])
    rows, cols = cost_map.shape

    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0

    vals = []
    while True:
        if not (0 <= r < rows and 0 <= c < cols):
            return float("inf")
        v = cost_map[r, c]
        if not np.isfinite(v):
            return float("inf")
        vals.append(float(v))
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc

    length = np.hypot(r1 - r0, c1 - c0)
    return (sum(vals) / len(vals)) * length


def theta_smooth_path(path, cost_map, cost_tol: float | None = None):
    """
    A* 결과 path를 Theta* 방식(any-angle)으로 후처리한다 — **비용-인지**.

        S -> 1 -> 2 -> 3 -> 4 -> G   (A* raw)
        S -> 3 -> G                  (smoothing)

    단축 채택 조건(둘 다 만족):
      1. 직선 경로상에 inf(통행불가) 없음
      2. 직선 세그먼트 비용 적분 ≤ 원본 구간 비용 × (1 + cost_tol)
    → 위험구역을 가로지르는 단축은 비용 증가로 거부되어, A*가 우회한 안전성을 보존한다.
    """
    if path is None:
        return None
    path = list(path)
    if len(path) <= 2:
        return path

    if cost_tol is None:
        cost_tol = getattr(config, "THETA_COST_TOL", 0.0)

    # 원본 경로의 누적 비용 (인접 waypoint 간 segment_cost 합)
    cum = [0.0]
    for k in range(1, len(path)):
        cum.append(cum[-1] + segment_cost(cost_map, path[k - 1], path[k]))

    smoothed = [path[0]]
    anchor_idx = 0
    i = 1
    while i < len(path):
        farthest = i
        for j in range(i + 1, len(path)):
            sc = segment_cost(cost_map, path[anchor_idx], path[j])
            orig = cum[j] - cum[anchor_idx]
            if np.isfinite(sc) and sc <= orig * (1.0 + cost_tol):
                farthest = j
            else:
                break
        smoothed.append(path[farthest])
        anchor_idx = farthest
        i = farthest + 1

    return smoothed


def _line_cells(a, b):
    """직선 a→b가 통과하는 셀 목록 (Bresenham)."""
    r0, c0 = int(a[0]), int(a[1])
    r1, c1 = int(b[0]), int(b[1])
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    cells = []
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc
    return cells


def traversed_cells(path):
    """waypoint 경로가 실제로 지나는 모든 셀 (직선 세그먼트 supercover, 끝점 중복 제거)."""
    if not path:
        return []
    full = []
    for a, b in zip(path[:-1], path[1:]):
        full += _line_cells(a, b)[:-1]
    full.append((int(path[-1][0]), int(path[-1][1])))
    return full


def path_stats(path, cost_map, slope_map, heightmap):
    """
    경로 통계. **waypoint가 아니라 실제 통과 셀** 기준(P6 수정).
      total_cost : 직선 세그먼트 비용 적분(segment_cost) 합 — 스무싱 게이트와 동일 척도.
      max_slope / elev : 통과 셀 전체 기준.
      distance_m : waypoint 간 직선거리 합.
    """
    cells = traversed_cells(path)

    total_cost = 0.0
    for a, b in zip(path[:-1], path[1:]):
        sc = segment_cost(cost_map, a, b)
        if np.isfinite(sc):
            total_cost += sc

    total_dist = sum(np.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(path[:-1], path[1:])) * config.GRID_RES

    heights = [float(heightmap[r, c]) for r, c in cells]
    max_slope = max(float(slope_map[r, c]) for r, c in cells)

    return {
        "waypoints": len(path),
        "traversed_cells": len(cells),
        "distance_m": total_dist,
        "total_cost": total_cost,
        "max_slope": max_slope,
        "elev_gain_m": max(0, max(heights) - heights[0]),
        "start_h_m": heights[0],
        "goal_h_m": heights[-1],
    }


def plan(start, goal, ts=None, cost_pref="risk", save=True, plot=True, apply_rocks=True):
    """
    전역 경로 계획 + 통계 + 시각화 + 저장.
      start, goal : (row, col) 그리드 좌표  (= 실제 m / GRID_RES)
      cost_pref   : "risk"(통합맵 우선) | "base"
    반환: (path, stats)  실패 시 (None, None)
    """
    bundle = mapio.load_maps(ts)
    cost_map, src = mapio.resolve_cost(bundle.ts, prefer=cost_pref)

    rock_mask = None
    los_surface = None
    if apply_rocks and getattr(config, "ENABLE_VIRTUAL_ROCKS", False):
        # cost_pref가 risk여도, 선택된 cost_map 위에 큰돌을 다시 주입한다.
        # 이 cost_map으로 A*를 재실행하므로 경로가 실제로 돌을 피해 바뀐다.
        cost_map, bundle.obstacle, rock_mask, los_surface = mapping.apply_virtual_rocks(
            cost_map,
            bundle.obstacle,
            bundle.heightmap_filled,
        )


    rows, cols = cost_map.shape

    print(f"\n=== A* + Theta smoothing 경로 탐색 ===")
    print(f"  출발: {start} ({start[0] * config.GRID_RES:.0f}m, {start[1] * config.GRID_RES:.0f}m)")
    print(f"  목적: {goal} ({goal[0] * config.GRID_RES:.0f}m, {goal[1] * config.GRID_RES:.0f}m)")
    print(f"  지도: {rows}×{cols}  비용맵: {src}")

    raw_path = astar(start, goal, cost_map, bundle.heightmap_filled)
    if raw_path is None:
        print("  [실패] 경로 없음")
        return None, None

    # ── Theta* 스타일 경로 스무싱 ─────────────────────────────
    # A*가 만든 격자형 경로에서 직선으로 연결 가능한 중간 waypoint를 제거한다.
    # cost=inf 구역은 Line-of-Sight 검사에서 차단되므로 돌과 통행불가 구역은 관통하지 않는다.
    path = theta_smooth_path(raw_path, cost_map)

    raw_st = path_stats(raw_path, cost_map, bundle.slope, bundle.heightmap_filled)
    st = path_stats(path, cost_map, bundle.slope, bundle.heightmap_filled)

    print(f"\n  ✅ 성공!")
    print(
        f"  [A* raw]       웨이포인트 {raw_st['waypoints']}  통과셀 {raw_st['traversed_cells']}  "
        f"거리 {raw_st['distance_m']:.1f}m  비용 {raw_st['total_cost']:.1f}  "
        f"최대경사 {raw_st['max_slope']:.1f}°"
    )
    print(
        f"  [Theta smooth] 웨이포인트 {st['waypoints']}  통과셀 {st['traversed_cells']}  "
        f"거리 {st['distance_m']:.1f}m  비용 {st['total_cost']:.1f}  "
        f"최대경사 {st['max_slope']:.1f}°  고도상승 {st['elev_gain_m']:.1f}m"
    )

    ts_now = mapio.timestamp()
    if plot:
        _plot(
            path,
            bundle,
            cost_map,
            start,
            goal,
            src,
            ts_now,
            rock_mask=rock_mask,
            raw_path=raw_path,
        )

    if save:
        raw_arr = np.array([[r * config.GRID_RES, c * config.GRID_RES] for r, c in raw_path])
        arr = np.array([[r * config.GRID_RES, c * config.GRID_RES] for r, c in path])

        # 기존 호환용: 최종 경로는 path_*.npy에 저장
        mapio.save_array(f"path_{ts_now}.npy", arr)

        # 비교용: 원본 A* / 스무싱 경로를 따로 저장
        mapio.save_array(f"path_raw_{ts_now}.npy", raw_arr)
        mapio.save_array(f"path_smooth_{ts_now}.npy", arr)

        print(f"[저장] path_{ts_now}.npy          {arr.shape}")
        print(f"[저장] path_raw_{ts_now}.npy      {raw_arr.shape}")
        print(f"[저장] path_smooth_{ts_now}.npy   {arr.shape}")

    return path, st


def plan_threat(start, goal, enemies, ts=None, w_surv=None,
                save=True, plot=True):
    """
    [Option S] 위협-인지 글로벌 계획.
      cost = base(돌 inf+마진 포함) + W_SURV_PLAN·hazard(채점기 생존체인)
      → A* 가 −log(생존)+지형 을 최소화 = 생존 최대화. + 비용-인지 Theta* 스무싱.

      start, goal : (row, col)
      enemies     : list[risk.Enemy]  (정적 — 미션당 precompute 1회)
    반환: (path, stats)
    """
    from . import eval as _eval   # 지연 import (순환참조 회피)

    bundle = mapio.load_maps(ts)
    base = np.array(bundle.cost, dtype=float, copy=True)

    rock_mask = None
    los_surface = None
    if getattr(config, "ENABLE_VIRTUAL_ROCKS", False):
        base, bundle.obstacle, rock_mask, los_surface = mapping.apply_virtual_rocks(
            base, bundle.obstacle, bundle.heightmap_filled)

    # 적-의존 위협장 (돌 높이를 LoS 차폐면으로 반영 → P4)
    precomp = _eval.precompute_threat(enemies, bundle, hm_los=los_surface)
    cost_map = _eval.threat_cost(bundle, precomp, w_surv=w_surv, base_cost=base)
    src = f"threat(W_SURV={w_surv if w_surv is not None else config.W_SURV_PLAN})"

    rows, cols = cost_map.shape
    print(f"\n=== [Option S] 위협-인지 A* + Theta smoothing ===")
    print(f"  출발: {start}  목적: {goal}  지도: {rows}×{cols}  비용: {src}")
    print(f"  적 {len(enemies)}기, 위협>0 셀 {int((precomp.intensity>0).sum())}")

    raw_path = astar(start, goal, cost_map, bundle.heightmap_filled)
    if raw_path is None:
        print("  [실패] 경로 없음")
        return None, None

    path = theta_smooth_path(raw_path, cost_map)   # 비용-인지 스무싱

    raw_st = path_stats(raw_path, cost_map, bundle.slope, bundle.heightmap_filled)
    st     = path_stats(path,     cost_map, bundle.slope, bundle.heightmap_filled)
    print(f"  [raw]   wp {raw_st['waypoints']:3d}  통과셀 {raw_st['traversed_cells']:3d}  "
          f"거리 {raw_st['distance_m']:.0f}m  비용 {raw_st['total_cost']:.1f}")
    print(f"  [smooth]wp {st['waypoints']:3d}  통과셀 {st['traversed_cells']:3d}  "
          f"거리 {st['distance_m']:.0f}m  비용 {st['total_cost']:.1f}")

    ts_now = mapio.timestamp()
    if plot:
        _plot_threat(path, bundle, cost_map, precomp, enemies, start, goal, src,
                     ts_now, rock_mask=rock_mask, raw_path=raw_path)
    if save:
        arr = np.array([[r * config.GRID_RES, c * config.GRID_RES] for r, c in path])
        mapio.save_array(f"path_{ts_now}.npy", arr)
        print(f"[저장] path_{ts_now}.npy  {arr.shape}")
    return path, st


def _plot_threat(path, b, cost_map, precomp, enemies, start, goal, src, ts,
                 rock_mask=None, raw_path=None):
    """Option S 전용 시각화: heightmap | threat intensity | threat_cost(적정 vmax)."""
    fig, axes = viz.plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f"Threat-aware Global Path (Option S)  {start}→{goal}  [{src}]", fontsize=12)
    viz.show(axes[0], b.heightmap_filled, "HeightMap (m)", "terrain", label="고도 (m)")
    viz.show(axes[1], precomp.intensity, "Threat intensity (0~1)", "inferno",
             vmin=0, vmax=1, label="0~1")
    finite = cost_map[np.isfinite(cost_map)]
    vmax = float(np.percentile(finite, 92)) if finite.size else 20.0
    viz.show(axes[2], cost_map, "threat_cost (계획 입력)", "inferno", vmax=vmax, label="비용")

    for ax in axes:
        for e in enemies:
            rs = [p[0] for p in e.positions]; cs = [p[1] for p in e.positions]
            if e.etype == "patrol":
                ax.plot(cs, rs, "c-", lw=2, alpha=.85)
            else:
                ax.plot(cs, rs, "cs", ms=9, mec="k")
        if getattr(config, "SHOW_VIRTUAL_ROCKS_ON_PLOT", True):
            viz.draw_rock_symbols(ax, label="rocks")
    if raw_path is not None and len(raw_path) > 0:
        rr = [p[0] for p in raw_path]; rc = [p[1] for p in raw_path]
        for ax in axes:
            ax.plot(rc, rr, color="gray", lw=1.0, ls="--", alpha=.7, label="A* raw")
    for ax in axes:
        viz.draw_path(ax, path)
    fig.tight_layout()
    out = viz.savefig(fig, f"global_path_{ts}.png")
    print(f"[저장] {out}")


def _plot(path, b, cost_map, start, goal, src, ts, rock_mask=None, raw_path=None):
    fig, axes = viz.plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f"Global Path (A* + Theta smoothing)  start={start} goal={goal}  [{src}]", fontsize=12)
    viz.show(axes[0], b.heightmap_filled, "HeightMap (m)", "terrain", label="고도 (m)")
    viz.show(axes[1], b.slope, "Slope (°)", "hot_r", vmax=45, label="경사도 (°)")
    viz.show(axes[2], cost_map, "Cost Map (계획 입력)", "RdYlGn_r", vmax=20, label="비용")

    if getattr(config, "SHOW_VIRTUAL_ROCKS_ON_PLOT", True):
        for ax in axes:
            viz.draw_rock_symbols(ax, label="virtual rocks")


    # 비교용: raw A* 경로를 얇은 회색 점선으로 표시
    # 기존 viz.draw_path()는 color/linewidth/linestyle 인자를 받지 않으므로
    # raw path는 여기서 직접 그린다.
    if raw_path is not None and len(raw_path) > 0:
        raw_rows = [p[0] for p in raw_path]
        raw_cols = [p[1] for p in raw_path]
        for ax in axes:
            ax.plot(
                raw_cols,
                raw_rows,
                color="gray",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
                label="A* raw",
            )

    # 최종 smoothing 경로
    for ax in axes:
        viz.draw_path(ax, path)

    fig.tight_layout()
    path_png = viz.savefig(fig, f"global_path_{ts}.png")
    print(f"[저장] {path_png}")
