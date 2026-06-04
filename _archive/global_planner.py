import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq
from datetime import datetime

# ── 타임스탬프 (최신 파일 기준) ─────────────────────────
TS = "20260530_150704"

# ── 맵 로드 ──────────────────────────────────────────────
cost_map    = np.load(f"cost_map_{TS}_final.npy")
obstacle    = np.load(f"obstacle_{TS}_final.npy")
slope_map   = np.load(f"slope_{TS}_final.npy")
heightmap   = np.load(f"heightmap_{TS}_filled_final.npy")

GRID_RES = 5.0   # 셀 하나 = 5m
ROWS, COLS = cost_map.shape

# ── A* 구현 ──────────────────────────────────────────────
def heuristic(a, b):
    """Euclidean 휴리스틱"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(r, c):
    """8방향 이웃 (대각선 포함)"""
    dirs = [(-1,0),(1,0),(0,-1),(0,1),
            (-1,-1),(-1,1),(1,-1),(1,1)]
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            yield nr, nc, np.sqrt(dr**2 + dc**2)  # 대각선 이동거리 √2

def astar(start, goal, cost_map):
    """
    A* 경로 탐색
    start, goal: (row, col) 튜플
    반환: path (list of (row,col)) 또는 None
    """
    if np.isinf(cost_map[start]):
        raise ValueError(f"출발지({start})가 통과 불가 구역입니다.")
    if np.isinf(cost_map[goal]):
        raise ValueError(f"목적지({goal})가 통과 불가 구역입니다.")

    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}
    g_score = np.full((ROWS, COLS), np.inf)
    g_score[start] = 0.0

    f_score = np.full((ROWS, COLS), np.inf)
    f_score[start] = heuristic(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # 경로 역추적
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        r, c = current
        for nr, nc, move_dist in get_neighbors(r, c):
            cell_cost = cost_map[nr, nc]
            if np.isinf(cell_cost):
                continue

            # 이동 비용 = 이동거리 × 셀 비용 + 고도차 패널티
            height_diff = abs(float(heightmap[nr, nc]) - float(heightmap[r, c]))
            tentative_g = g_score[r, c] + move_dist * cell_cost + height_diff * 0.1

            if tentative_g < g_score[nr, nc]:
                came_from[(nr, nc)] = current
                g_score[nr, nc] = tentative_g
                f_score[nr, nc] = tentative_g + heuristic((nr, nc), goal)
                heapq.heappush(open_set, (f_score[nr, nc], (nr, nc)))

    return None  # 경로 없음

# ── 경로 통계 계산 ────────────────────────────────────────
def path_stats(path):
    total_dist = 0.0
    total_cost = 0.0
    max_slope  = 0.0
    heights    = []

    for i in range(len(path)):
        r, c = path[i]
        heights.append(float(heightmap[r, c]))
        total_cost += float(cost_map[r, c])
        max_slope   = max(max_slope, float(slope_map[r, c]))
        if i > 0:
            pr, pc = path[i-1]
            dr, dc = r - pr, c - pc
            total_dist += np.sqrt(dr**2 + dc**2) * GRID_RES

    elev_gain = max(0, max(heights) - heights[0])
    elev_loss = max(0, heights[0] - min(heights))

    return {
        "waypoints"  : len(path),
        "distance_m" : total_dist,
        "total_cost" : total_cost,
        "max_slope"  : max_slope,
        "elev_gain_m": elev_gain,
        "start_h_m"  : heights[0],
        "goal_h_m"   : heights[-1],
    }

# ── 시각화 ────────────────────────────────────────────────
def visualize(path, start, goal, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f"Global Path (A*)  {title_suffix}", fontsize=13)

    maps   = [heightmap, slope_map, cost_map]
    cmaps  = ["terrain", "hot_r", "RdYlGn_r"]
    titles = ["HeightMap (m)", "Slope Map (°)", "Cost Map"]
    vmaxs  = [None, 45, 20]

    for ax, data, cm, title, vm in zip(axes, maps, cmaps, titles, vmaxs):
        finite = data[np.isfinite(data)]
        vmax   = vm if vm else (float(finite.max()) if len(finite) else 1)
        im     = ax.imshow(np.where(np.isinf(data), np.nan, data),
                           origin="lower", cmap=cm, aspect="equal", vmax=vmax)
        # inf = 장애물: 검정으로 표시
        inf_mask = np.isinf(data)
        if inf_mask.any():
            ax.imshow(np.where(inf_mask, 1, np.nan),
                      origin="lower", cmap="gray", aspect="equal",
                      alpha=0.8, vmin=0, vmax=1)

        if path:
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            ax.plot(xs, ys, "b-", linewidth=2, label="path")
            ax.plot(xs[0],  ys[0],  "go", markersize=10, label="start")
            ax.plot(xs[-1], ys[-1], "r*", markersize=12, label="goal")
            ax.legend(fontsize=8, loc="upper right")

        ax.set_title(title)
        ax.set_xlabel("X (cell)")
        ax.set_ylabel("Z (cell)")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname  = f"global_path_{ts_now}2.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[저장] {fname}")

# ── 메인 ─────────────────────────────────────────────────
def plan(start_rc, goal_rc):
    """
    start_rc, goal_rc: (row, col)  ← 그리드 좌표
    grid 좌표 = 실제 좌표(m) / 5
    예) 실제 (50m, 100m) → (10, 20)
    """
    print(f"\n=== A* 경로 탐색 ===")
    print(f"  출발: {start_rc}  ({start_rc[0]*GRID_RES:.0f}m, {start_rc[1]*GRID_RES:.0f}m)")
    print(f"  목적: {goal_rc}  ({goal_rc[0]*GRID_RES:.0f}m, {goal_rc[1]*GRID_RES:.0f}m)")
    print(f"  지도 크기: {ROWS}×{COLS}  해상도: {GRID_RES}m/cell")

    path = astar(start_rc, goal_rc, cost_map)

    if path is None:
        print("  [실패] 경로를 찾을 수 없습니다.")
        return None

    stats = path_stats(path)
    print(f"\n  경로 탐색 성공!")
    print(f"  - 웨이포인트 수 : {stats['waypoints']}")
    print(f"  - 총 거리       : {stats['distance_m']:.1f} m")
    print(f"  - 총 비용       : {stats['total_cost']:.2f}")
    print(f"  - 최대 경사     : {stats['max_slope']:.1f}°")
    print(f"  - 고도 상승     : {stats['elev_gain_m']:.1f} m")
    print(f"  - 출발 고도     : {stats['start_h_m']:.1f} m")
    print(f"  - 도착 고도     : {stats['goal_h_m']:.1f} m")

    suffix = f"start={start_rc} goal={goal_rc}"
    visualize(path, start_rc, goal_rc, title_suffix=suffix)

    # 경로 좌표 저장 (m 단위)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_arr = np.array([[r * GRID_RES, c * GRID_RES] for r, c in path])
    np.save(f"path_{ts_now}.npy", path_arr)
    print(f"[저장] path_{ts_now}.npy  (shape: {path_arr.shape})")

    return path, stats


if __name__ == "__main__":
    # ── 출발/목적지 설정 (그리드 셀 단위) ─────────────────
    # 맵 크기: 61×61 (0~60)
    # 실제 거리: 0~300m (61셀 × 5m)
    #
    # 예: 좌하단(5,5) → 우상단(55,55)
    START = (5, 5)
    GOAL  = (5, 55)

    plan(START, GOAL)
