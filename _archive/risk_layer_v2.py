import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from datetime import datetime

# ══════════════════════════════════════════════════════════
#  K2 전차 기반 리스크 레이어 v2
#  v1 (능선 + 개활지) + LoS (적 가시선)
#
#  적 입력 형식 (터미널):
#    고정 보병 진지 : infantry  x z
#    고정 전차      : tank      x z
#    순찰 전차      : patrol    x1 z1  x2 z2  x3 z3 ...
#
#  좌표는 실제 미터(m) 단위 → 내부에서 셀로 변환
# ══════════════════════════════════════════════════════════

# ── 설정 ──────────────────────────────────────────────────
TS       = "20260530_150704"
GRID_RES = 5.0

# K2 제원
K2_HEIGHT_M = 2.4
K2_WIDTH_M  = 3.6

# 능선 리스크
RIDGE_RADIUS_CELLS = 4
RIDGE_EXPOSE_THR   = K2_HEIGHT_M

# 개활지 리스크
OPEN_RADIUS_CELLS = 6
OPEN_MIN_COVER    = 0.05

# ── 적 탐지 범위 (m) ──────────────────────────────────────
DETECT_RANGE = {
    "infantry": 500,   # 보병 진지: 500m
    "tank":    2000,   # 전차 (열상): 2000m
    "patrol":  2000,   # 순찰 전차도 동일
}

# ── 관측 높이 (지면 위) ───────────────────────────────────
OBS_HEIGHT = {
    "infantry":  1.8,  # 보병 눈높이 (m)
    "tank":      2.4,  # 전차 전고 = 열상 센서 높이 (m)
    "patrol":    2.4,
}

# ── 가중치 ────────────────────────────────────────────────
W_BASE  = 1.0
W_RIDGE = 3.0
W_OPEN  = 2.0
W_LOS   = 5.0   # LoS — 직접 피탐지는 가장 치명적


# ══════════════════════════════════════════════════════════
#  v1 리스크 함수 (그대로 유지)
# ══════════════════════════════════════════════════════════

def make_ridge_risk(hm):
    kernel     = 2 * RIDGE_RADIUS_CELLS + 1
    local_mean = uniform_filter(hm, size=kernel)
    height_above = hm - local_mean
    ridge_raw  = np.clip(height_above / RIDGE_EXPOSE_THR, 0.0, None)
    return np.clip(ridge_raw / 2.0, 0.0, 1.0)


def make_open_risk(obstacle, cost_map):
    cover_mask = obstacle.astype(float)
    cover_mask[np.isinf(cost_map)] = 1.0
    kernel        = 2 * OPEN_RADIUS_CELLS + 1
    cover_density = uniform_filter(cover_mask, size=kernel)
    return 1.0 - np.clip(
        (cover_density - OPEN_MIN_COVER) / (0.20 - OPEN_MIN_COVER), 0.0, 1.0
    )


# ══════════════════════════════════════════════════════════
#  LoS Ray Casting
# ══════════════════════════════════════════════════════════

def bresenham(r0, c0, r1, c1):
    """Bresenham 직선 알고리즘 — 두 셀 사이의 셀 목록 반환"""
    cells = []
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc
    return cells


def is_visible(hm, src_r, src_c, src_h_offset,
               tgt_r, tgt_c, tgt_h_offset=K2_HEIGHT_M):
    """
    src → tgt 가시선 체크 (지형 차폐 고려)
    src_h_offset : 관측자 눈높이 (지면 위 m)
    tgt_h_offset : 피관측체 높이 (K2 전고 기본)
    """
    cells = bresenham(src_r, src_c, tgt_r, tgt_c)
    if len(cells) < 2:
        return True

    src_elev = float(hm[src_r, src_c]) + src_h_offset
    tgt_elev = float(hm[tgt_r, tgt_c]) + tgt_h_offset
    total    = len(cells) - 1

    for i, (r, c) in enumerate(cells[1:-1], start=1):
        # 직선 보간 고도
        t         = i / total
        los_elev  = src_elev + t * (tgt_elev - src_elev)
        ground_h  = float(hm[r, c])
        if ground_h > los_elev:
            return False   # 지형이 시선을 차단

    return True


def make_los_risk(hm, enemies, rows, cols):
    """
    enemies: list of dict
      { "type": "infantry"|"tank"|"patrol",
        "positions": [(r,c), ...]   ← 고정=1개, 순찰=경로 전체 }
    반환: los_risk (0~1, rows×cols)
    """
    los_acc = np.zeros((rows, cols), dtype=float)   # 누적 가시 횟수

    total_positions = sum(len(e["positions"]) for e in enemies)
    done = 0

    for enemy in enemies:
        etype     = enemy["type"]
        det_range = DETECT_RANGE[etype]
        obs_h     = OBS_HEIGHT[etype]
        det_cells = int(det_range / GRID_RES)

        for (er, ec) in enemy["positions"]:
            done += 1
            print(f"  LoS 계산 중... ({done}/{total_positions}) "
                  f"type={etype} pos=({er},{ec})", end="\r")

            # 탐지 범위 내 셀만 체크 (전체 맵 체크하면 너무 느림)
            r_min = max(0, er - det_cells)
            r_max = min(rows, er + det_cells + 1)
            c_min = max(0, ec - det_cells)
            c_max = min(cols, ec + det_cells + 1)

            for r in range(r_min, r_max):
                for c in range(c_min, c_max):
                    dist_m = np.sqrt((r-er)**2 + (c-ec)**2) * GRID_RES
                    if dist_m > det_range:
                        continue
                    if is_visible(hm, er, ec, obs_h, r, c):
                        # 거리 감쇠: 멀수록 리스크 낮음
                        decay = 1.0 - (dist_m / det_range) ** 0.5
                        los_acc[r, c] += decay

    print()  # 줄바꿈

    # 정규화 (누적 횟수 → 0~1)
    if los_acc.max() > 0:
        los_risk = np.clip(los_acc / los_acc.max(), 0.0, 1.0)
    else:
        los_risk = los_acc

    return los_risk


# ══════════════════════════════════════════════════════════
#  터미널 입력 파서
# ══════════════════════════════════════════════════════════

def parse_enemies(rows, cols):
    """
    터미널에서 적 정보 입력받기
    형식:
      infantry x z          ← 보병 진지 (m 단위)
      tank     x z          ← 고정 전차 (m 단위)
      patrol   x1 z1 x2 z2 ...  ← 순찰 전차 웨이포인트 (m 단위)
      done                  ← 입력 종료
    """
    print("\n" + "="*55)
    print("  적 위치 입력  (좌표: 실제 미터(m) 단위)")
    print("  맵 범위: X 0~{:.0f}m  Z 0~{:.0f}m".format(
        (cols-1)*GRID_RES, (rows-1)*GRID_RES))
    print("-"*55)
    print("  명령어:")
    print("    infantry x z           → 보병 진지 (탐지 500m)")
    print("    tank     x z           → 고정 전차 (탐지 2000m)")
    print("    patrol   x1 z1 x2 z2  → 순찰 전차 (웨이포인트)")
    print("    done                   → 입력 완료")
    print("  예시:")
    print("    infantry 150 200")
    print("    tank 250 50")
    print("    patrol 100 100 200 100 200 200 100 200")
    print("="*55)

    enemies = []

    while True:
        try:
            raw = input("\n적 입력 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        tokens = raw.split()
        cmd    = tokens[0].lower()

        if cmd == "done":
            break

        if cmd not in ("infantry", "tank", "patrol"):
            print("  ❌ 명령어 오류. infantry / tank / patrol / done 중 입력하세요.")
            continue

        nums = tokens[1:]

        # 좌표 파싱
        try:
            coords = [float(v) for v in nums]
        except ValueError:
            print("  ❌ 숫자 파싱 실패. 좌표는 숫자로 입력하세요.")
            continue

        if len(coords) < 2 or len(coords) % 2 != 0:
            print("  ❌ 좌표는 x z 쌍으로 입력하세요. (짝수 개)")
            continue

        # m → 셀 변환 + 범위 클램프
        positions = []
        for i in range(0, len(coords), 2):
            x_m, z_m = coords[i], coords[i+1]
            r = int(round(z_m / GRID_RES))   # row = Z축
            c = int(round(x_m / GRID_RES))   # col = X축
            r = max(0, min(rows-1, r))
            c = max(0, min(cols-1, c))
            positions.append((r, c))

        # 순찰의 경우 경로 보간 (웨이포인트 사이 셀 채우기)
        if cmd == "patrol" and len(positions) > 1:
            from itertools import pairwise
            full_path = []
            for (r0,c0), (r1,c1) in zip(positions[:-1], positions[1:]):
                full_path.extend(bresenham(r0, c0, r1, c1)[:-1])
            full_path.append(positions[-1])
            positions = full_path
            print(f"  ✅ patrol 등록: 웨이포인트 {len(coords)//2}개 "
                  f"→ 경로 셀 {len(positions)}개")
        else:
            print(f"  ✅ {cmd} 등록: 셀 위치 {positions}")

        enemies.append({"type": cmd, "positions": positions})

    if not enemies:
        print("  ⚠️  적 정보 없음 — LoS 리스크 = 0")

    return enemies


# ══════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════

def main():
    print(f"[로드] {TS} 맵 파일...")
    hm       = np.load(f"heightmap_{TS}_filled_final.npy")
    obstacle = np.load(f"obstacle_{TS}_final.npy")
    cost_v3  = np.load(f"cost_map_{TS}_final.npy")
    slope    = np.load(f"slope_{TS}_final.npy")

    ROWS, COLS = hm.shape
    print(f"  맵 크기: {ROWS}×{COLS}  ({ROWS*GRID_RES:.0f}m × {COLS*GRID_RES:.0f}m)")

    # ── 적 입력 ───────────────────────────────────────────
    enemies = parse_enemies(ROWS, COLS)

    # ── 리스크 계산 ───────────────────────────────────────
    print("\n[계산] 능선 리스크...")
    ridge_risk = make_ridge_risk(hm)

    print("[계산] 개활지 리스크...")
    open_risk  = make_open_risk(obstacle, cost_v3)

    print("[계산] LoS 리스크...")
    los_risk   = make_los_risk(hm, enemies, ROWS, COLS)

    print("[계산] 통합 코스트...")
    combined = (W_BASE  * np.where(np.isinf(cost_v3), 0, cost_v3)
              + W_RIDGE * ridge_risk
              + W_OPEN  * open_risk
              + W_LOS   * los_risk)
    combined[np.isinf(cost_v3)] = np.inf

    # ── 통계 ─────────────────────────────────────────────
    passable = ~np.isinf(combined)
    print(f"\n[통계]")
    print(f"  능선 리스크 — 평균: {ridge_risk[passable].mean():.3f}  최대: {ridge_risk[passable].max():.3f}")
    print(f"  개활지 리스크 — 평균: {open_risk[passable].mean():.3f}  최대: {open_risk[passable].max():.3f}")
    print(f"  LoS 리스크  — 평균: {los_risk[passable].mean():.3f}  최대: {los_risk[passable].max():.3f}")
    print(f"  통합 코스트 범위: {combined[passable].min():.2f} ~ {np.percentile(combined[passable],95):.2f}")

    # ── 저장 ─────────────────────────────────────────────
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f"ridge_risk_{ts_now}.npy",   ridge_risk)
    np.save(f"open_risk_{ts_now}.npy",    open_risk)
    np.save(f"los_risk_{ts_now}.npy",     los_risk)
    np.save(f"cost_risk_v2_{ts_now}.npy", combined)
    print(f"\n[저장] 리스크 레이어 + cost_risk_v2 — {ts_now}")

    # ── 시각화 ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    fig.suptitle(f"K2 전차 리스크 레이어 v2 (LoS 포함)  {ts_now}", fontsize=13)

    def show(ax, data, title, cmap, vmin=None, vmax=None, label=""):
        finite = data[np.isfinite(data)]
        vm = vmax if vmax is not None else (float(finite.max()) if len(finite) else 1)
        im = ax.imshow(np.where(np.isinf(data), np.nan, data),
                       origin="lower", cmap=cmap, aspect="equal",
                       vmin=vmin, vmax=vm)
        if np.isinf(data).any():
            ax.imshow(np.where(np.isinf(data), 1, np.nan),
                      origin="lower", cmap="gray", aspect="equal",
                      alpha=0.9, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (cell)"); ax.set_ylabel("Z (cell)")
        plt.colorbar(im, ax=ax, label=label)

    # 적 위치 마커 표시
    def mark_enemies(ax):
        markers = {"infantry": ("b^", 10, "보병"), 
                   "tank":     ("rs", 12, "전차"),
                   "patrol":   ("m.", 6,  "순찰")}
        for e in enemies:
            m, ms, label = markers[e["type"]]
            if e["type"] == "patrol":
                # 순찰은 경로만 표시 (점 너무 많아서)
                rs = [p[0] for p in e["positions"]]
                cs = [p[1] for p in e["positions"]]
                ax.plot(cs, rs, "m-", linewidth=1.5, alpha=0.7, label=f"순찰경로")
                ax.plot(cs[0], rs[0], "m^", markersize=10)
            else:
                for (r, c) in e["positions"]:
                    ax.plot(c, r, m, markersize=ms, label=label)

    show(axes[0,0], hm,         "HeightMap (m)",       "terrain",  label="고도 (m)")
    show(axes[0,1], ridge_risk, "Ridge Risk (능선)",    "YlOrRd",   vmin=0, vmax=1, label="0~1")
    show(axes[0,2], open_risk,  "Open Risk (개활지)",   "YlOrRd",   vmin=0, vmax=1, label="0~1")
    show(axes[1,0], los_risk,   "LoS Risk (적 가시선)", "YlOrRd",   vmin=0, vmax=1, label="0~1")
    show(axes[1,1], cost_v3,    "Base Cost Map",        "RdYlGn_r", vmax=20, label="비용")
    show(axes[1,2], combined,   "Combined Cost v2\n(A* 입력용)", "RdYlGn_r", label="통합 비용")

    for ax in axes.flat:
        mark_enemies(ax)

    # 범례 중복 제거
    handles, labels = axes[1,2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        axes[1,2].legend(by_label.values(), by_label.keys(),
                         fontsize=8, loc="upper right")

    plt.tight_layout()
    fname = f"risk_layer_v2_{ts_now}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {fname}")

    print(f"\n✅ 완료!")
    print(f"   global_planner.py 에서 사용할 파일:")
    print(f"   cost_risk_v2_{ts_now}.npy")

    return combined, ts_now


if __name__ == "__main__":
    main()
