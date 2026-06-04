"""
[리스크 레이어]  ridge + open + LoS   (구 risklayer.py / risk_layer_v2.py 통합)

  Ridge Risk : 능선 노출 (피탐지)
  Open Risk  : 개활지 노출 (엄폐 부족)
  LoS Risk   : 적 가시선 직접 노출 (ray casting)

적 정보는 Enemy 리스트로 받음 (터미널 입력 parse_enemies 또는 코드에서 직접).
사격통제(fire_control)·비전(perception)이 붙으면 적 위치를 자동 공급하는
형태로 확장 예정.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import uniform_filter

from . import config, mapio, viz, mapping


# ══════════════════════════════════════════════════════════
#  적 모델
# ══════════════════════════════════════════════════════════
@dataclass
class Enemy:
    etype: str                       # "infantry" | "tank" | "patrol"
    positions: list[tuple[int, int]] = field(default_factory=list)  # [(row,col), ...]


# ══════════════════════════════════════════════════════════
#  지형 기반 리스크 (적 정보 불필요)
# ══════════════════════════════════════════════════════════
def make_ridge_risk(hm: np.ndarray) -> np.ndarray:
    """능선 노출: 주변 평균보다 K2 전고 이상 높으면 노출 (0~1)"""
    kernel       = 2 * config.RIDGE_RADIUS_CELLS + 1
    local_mean   = uniform_filter(hm, size=kernel)
    height_above = hm - local_mean
    ridge_raw    = np.clip(height_above / config.RIDGE_EXPOSE_THR, 0.0, None)
    return np.clip(ridge_raw / 2.0, 0.0, 1.0)   # 2배(4.8m)↑ 면 최대


def make_open_risk(obstacle: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """개활지 노출: 반경 내 엄폐 밀도가 낮을수록 리스크↑ (0~1)"""
    cover = obstacle.astype(float)
    cover[np.isinf(cost)] = 1.0
    kernel  = 2 * config.OPEN_RADIUS_CELLS + 1
    density = uniform_filter(cover, size=kernel)
    span    = config.OPEN_MAX_COVER - config.OPEN_MIN_COVER
    return 1.0 - np.clip((density - config.OPEN_MIN_COVER) / span, 0.0, 1.0)


# ══════════════════════════════════════════════════════════
#  LoS Ray Casting
# ══════════════════════════════════════════════════════════
def bresenham(r0, c0, r1, c1):
    """두 셀 사이 직선 셀 목록"""
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
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


def is_visible(hm, src_r, src_c, src_h, tgt_r, tgt_c,
               tgt_h=config.K2_HEIGHT_M) -> bool:
    """지형 차폐를 고려한 src→tgt 가시선 판정"""
    cells = bresenham(src_r, src_c, tgt_r, tgt_c)
    if len(cells) < 2:
        return True
    src_elev = float(hm[src_r, src_c]) + src_h
    tgt_elev = float(hm[tgt_r, tgt_c]) + tgt_h
    total    = len(cells) - 1
    for i, (r, c) in enumerate(cells[1:-1], start=1):
        los_elev = src_elev + (i / total) * (tgt_elev - src_elev)
        if float(hm[r, c]) > los_elev:
            return False
    return True


def make_los_risk(hm: np.ndarray, enemies: list[Enemy]) -> np.ndarray:
    """적 가시선 누적 → 0~1 정규화"""
    rows, cols = hm.shape
    los = np.zeros((rows, cols), dtype=float)

    total = sum(len(e.positions) for e in enemies)
    done = 0
    for e in enemies:
        det_range = config.DETECT_RANGE[e.etype]
        obs_h     = config.OBS_HEIGHT[e.etype]
        det_cells = int(det_range / config.GRID_RES)
        for (er, ec) in e.positions:
            done += 1
            print(f"  LoS... ({done}/{total}) {e.etype} ({er},{ec})", end="\r")
            r0, r1 = max(0, er - det_cells), min(rows, er + det_cells + 1)
            c0, c1 = max(0, ec - det_cells), min(cols, ec + det_cells + 1)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    dist_m = np.sqrt((r-er)**2 + (c-ec)**2) * config.GRID_RES
                    if dist_m > det_range:
                        continue
                    if is_visible(hm, er, ec, obs_h, r, c):
                        los[r, c] += 1.0 - (dist_m / det_range) ** 0.5
    if total:
        print()
    return np.clip(los / los.max(), 0.0, 1.0) if los.max() > 0 else los


# ══════════════════════════════════════════════════════════
#  통합 코스트
# ══════════════════════════════════════════════════════════
def combine(base_cost, ridge, open_, los=None):
    """가중합 통합 코스트. inf(통과불가) 셀 보존."""
    finite_base = np.where(np.isinf(base_cost), 0, base_cost)
    combined = (config.W_BASE  * finite_base
              + config.W_RIDGE * ridge
              + config.W_OPEN  * open_)
    if los is not None:
        combined += config.W_LOS * los
    combined[np.isinf(base_cost)] = np.inf
    return combined


def build_risk_cost(bundle: mapio.MapBundle | None = None,
                    enemies: list[Enemy] | None = None,
                    save: bool = True, plot: bool = True) -> np.ndarray:
    """
    리스크 통합 코스트 생성 + 저장.
    enemies=None 이면 LoS 생략(지형 리스크만 = v1 동작).

    가상 큰돌이 켜져 있으면 여기서 먼저 bundle.cost / bundle.obstacle에 반영한다.
    그래서 risk_layer.png의 Open Risk와 Combined Cost도 큰돌을 반영한다.
    """
    if bundle is None:
        bundle = mapio.load_maps()

    hm = bundle.heightmap_filled
    rock_mask = None
    cover_mask = None
    los_surface = None

    if getattr(config, "ENABLE_VIRTUAL_ROCKS", False):
        # 중요: risk 계산 전에 먼저 돌을 obstacle/cost에 반영해야 open risk가 바뀜.
        bundle.cost, bundle.obstacle, rock_mask, los_surface = mapping.apply_virtual_rocks(
            bundle.cost,
            bundle.obstacle,
            hm,
        )

    print("[계산] 능선 리스크...")
    ridge = make_ridge_risk(hm)

    print("[계산] 개활지 리스크...")
    open_ = make_open_risk(bundle.obstacle, bundle.cost)

    # 큰돌 주변은 엄폐 가능 구역이므로 open risk를 낮춤.
    if getattr(config, "ENABLE_VIRTUAL_ROCKS", False):
        open_, rock_mask2, cover_mask = mapping.apply_virtual_rock_open_risk(open_, hm.shape)
        if rock_mask is None:
            rock_mask = rock_mask2

    enemies_for_los = list(enemies or [])

    los = None
    hm_los = los_surface if (los_surface is not None and getattr(config, "ROCK_USE_LOS_SURFACE", True)) else hm

    if enemies_for_los:
        print("[계산] LoS 리스크...")
        # 돌 높이를 LoS 차폐면에 반영. 지형 높이 자체는 바꾸지 않고 시야 계산에만 사용.
        los = make_los_risk(hm_los, enemies_for_los)

    combined = combine(bundle.cost, ridge, open_, los)

    ver = "v2" if enemies else "v1"
    ts  = mapio.timestamp()
    if save:
        mapio.save_array(f"ridge_risk_{ts}.npy", ridge)
        mapio.save_array(f"open_risk_{ts}.npy",  open_)
        if los is not None:
            mapio.save_array(f"los_risk_{ts}.npy", los)
        if rock_mask is not None:
            mapio.save_array(f"rock_mask_{ts}.npy", rock_mask.astype(bool))
        if cover_mask is not None:
            mapio.save_array(f"rock_cover_mask_{ts}.npy", cover_mask.astype(bool))
        mapio.save_array(f"cost_risk_{ver}_{ts}.npy", combined)
        print(f"[저장] cost_risk_{ver}_{ts}.npy")

    if plot:
        _plot(bundle, ridge, open_, los, combined, enemies_for_los, ts,
              rock_mask=rock_mask, cover_mask=cover_mask)

    return combined


def _plot(b, ridge, open_, los, combined, enemies, ts, rock_mask=None, cover_mask=None):
    fig, axes = viz.plt.subplots(2, 3, figsize=(21, 14))
    fig.suptitle(f"K2 리스크 레이어 ({'LoS포함' if los is not None else '지형'})  {ts}",
                 fontsize=13)
    viz.show(axes[0,0], b.heightmap_filled, "HeightMap (m)", "terrain", label="고도 (m)")
    viz.show(axes[0,1], ridge, "Ridge Risk (능선)", "YlOrRd", vmin=0, vmax=1, label="0~1")
    viz.show(axes[0,2], open_, "Open Risk (개활지)", "YlOrRd", vmin=0, vmax=1, label="0~1")
    if los is not None:
        viz.show(axes[1,0], los, "LoS Risk (적가시선)", "YlOrRd", vmin=0, vmax=1, label="0~1")
    else:
        axes[1,0].set_title("LoS Risk (적 없음)"); axes[1,0].axis("off")
    viz.show(axes[1,1], b.cost, "Base Cost", "RdYlGn_r", vmax=20, label="비용")
    viz.show(axes[1,2], combined, "Combined Cost (A* 입력)", "RdYlGn_r", label="통합 비용")

    # 큰돌/엄폐 영향권 오버레이
    if getattr(config, "SHOW_VIRTUAL_ROCKS_ON_PLOT", True):
        for ax in axes.flat:
            viz.draw_rock_cover_zones(ax, label="rock cover")
            viz.draw_rock_symbols(ax, label="virtual rocks")


    # 적 마커
    markers = {"infantry": ("b^", 10), "tank": ("rs", 12), "patrol": ("m.", 6)}
    for ax in axes.flat:
        for e in enemies:
            m, ms = markers[e.etype]
            if e.etype == "patrol":
                rs = [p[0] for p in e.positions]; cs = [p[1] for p in e.positions]
                ax.plot(cs, rs, "m-", linewidth=1.5, alpha=0.7)
                ax.plot(cs[0], rs[0], "m^", markersize=10)
            else:
                for (r, c) in e.positions:
                    ax.plot(c, r, m, markersize=ms)
    fig.tight_layout()
    path = viz.savefig(fig, f"risk_layer_{ts}.png")
    print(f"[저장] {path}")


# ══════════════════════════════════════════════════════════
#  터미널 입력 파서
# ══════════════════════════════════════════════════════════
def parse_enemies(rows: int, cols: int) -> list[Enemy]:
    """
    터미널에서 적 위치 입력 (좌표: 실제 m).
      infantry x z          / tank x z
      patrol   x1 z1 x2 z2 ...
      done
    """
    g = config.GRID_RES
    print("\n" + "="*55)
    print("  적 위치 입력 (좌표: 실제 m)")
    print(f"  맵 범위: X 0~{(cols-1)*g:.0f}m  Z 0~{(rows-1)*g:.0f}m")
    print("  infantry x z | tank x z | patrol x1 z1 x2 z2 ... | done")
    print("="*55)

    enemies: list[Enemy] = []
    while True:
        try:
            raw = input("\n적 입력 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        tok = raw.split()
        cmd = tok[0].lower()
        if cmd == "done":
            break
        if cmd not in ("infantry", "tank", "patrol"):
            print("  ❌ infantry / tank / patrol / done 중 입력")
            continue
        try:
            coords = [float(v) for v in tok[1:]]
        except ValueError:
            print("  ❌ 좌표는 숫자로")
            continue
        if len(coords) < 2 or len(coords) % 2:
            print("  ❌ x z 쌍(짝수 개)으로 입력")
            continue

        pos = []
        for i in range(0, len(coords), 2):
            r = max(0, min(rows-1, int(round(coords[i+1] / g))))  # row = Z
            c = max(0, min(cols-1, int(round(coords[i]   / g))))  # col = X
            pos.append((r, c))

        if cmd == "patrol" and len(pos) > 1:
            full = []
            for (r0,c0), (r1,c1) in zip(pos[:-1], pos[1:]):
                full.extend(bresenham(r0, c0, r1, c1)[:-1])
            full.append(pos[-1])
            pos = full
            print(f"  ✅ patrol: 경로 셀 {len(pos)}개")
        else:
            print(f"  ✅ {cmd}: {pos}")
        enemies.append(Enemy(cmd, pos))

    if not enemies:
        print("  ⚠️  적 없음 — LoS 생략")
    return enemies
