"""
[수집 후처리] 보정 + Base Cost 생성   (구 mapv3.py 통합)

Yaxis.py 가 수집한 raw 맵(heightmap/obstacle/filled)을 받아:
  1. 전치 보정 (correct)
  2. 경사도 + 절벽 + 장애물팽창 → cost v3
  3. *_final.npy 세트로 저장

비전(perception.py)에서 물/숲/바위 마스크가 생기면
make_cost() 안에서 합산하도록 확장 예정.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import (binary_dilation, gaussian_filter,
                           maximum_filter, minimum_filter)

from . import config, mapio, viz


def correct(arr: np.ndarray) -> np.ndarray:
    """최종 좌표 보정: 전치(Transpose)"""
    return arr.T


def make_cost(hm: np.ndarray, obs: np.ndarray,
              extra_inf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    경사도 + 비용맵 (v3) 생성.

    hm        : 보정된 보간 heightmap
    obs       : 보정된 장애물 bool
    extra_inf : (옵션) 추가로 통과불가 처리할 bool 마스크
                — 비전 기반 물/깊은수심 등을 여기로 주입할 자리

    반환: (slope_deg, cost)
    """
    hm_smooth = gaussian_filter(hm, sigma=1.0)
    gx, gz    = np.gradient(hm_smooth, config.GRID_RES)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gz**2)))

    cost = np.ones_like(slope_deg)
    for thr, val in config.SLOPE_COST:
        cost[slope_deg > thr] = val

    # 절벽: 3x3 내 고도차
    diff = maximum_filter(hm, size=3) - minimum_filter(hm, size=3)
    cost[diff > config.CLIFF_DIFF_M] = np.inf

    # 장애물
    cost[obs.astype(bool)] = np.inf

    # 외부 주입 통과불가 (비전 등)
    if extra_inf is not None:
        cost[extra_inf.astype(bool)] = np.inf

    # 장애물 팽창 (inflation)
    inflated = binary_dilation(np.isinf(cost), iterations=config.INFLATE_ITERS)
    cost[inflated & ~np.isinf(cost)] = config.INFLATE_COST

    return slope_deg, cost


def build_final_maps(raw_ts: str, save: bool = True,
                     plot: bool = True) -> mapio.MapBundle:
    """
    raw_ts 의 수집본을 보정·cost 생성하여 새 _final 세트로 저장.
    반환: 새로 만든 MapBundle (새 타임스탬프)
    """
    d = config.DATA_DIR
    hm_raw  = np.load(d / f"heightmap_{raw_ts}.npy")
    hm_fill = np.load(d / f"heightmap_{raw_ts}_filled.npy")
    obs     = np.load(d / f"obstacle_{raw_ts}.npy")

    hm_raw_c  = correct(hm_raw)
    hm_fill_c = correct(hm_fill)
    obs_c     = correct(obs)

    slope, cost = make_cost(hm_fill_c, obs_c)

    ts = mapio.timestamp()
    bundle = mapio.MapBundle(
        ts=ts, heightmap=hm_raw_c, heightmap_filled=hm_fill_c,
        slope=slope, cost=cost, obstacle=obs_c,
    )

    if save:
        mapio.save_array(f"heightmap_{ts}_final.npy",        hm_raw_c)
        mapio.save_array(f"heightmap_{ts}_filled_final.npy", hm_fill_c)
        mapio.save_array(f"slope_{ts}_final.npy",            slope)
        mapio.save_array(f"cost_map_{ts}_final.npy",         cost)
        mapio.save_array(f"obstacle_{ts}_final.npy",         obs_c)
        print(f"✅ 최종 맵 저장 — {ts}")

    if plot:
        _plot(bundle)

    return bundle


def _plot(b: mapio.MapBundle):
    fig, axes = viz.plt.subplots(1, 3, figsize=(20, 6))
    viz.show(axes[0], b.heightmap_filled, "HeightMap (최종)", "terrain", label="고도 (m)")
    viz.show(axes[1], b.slope, "Slope Map v3", "hot_r", vmin=0, vmax=45, label="경사도 (°)")
    viz.show(axes[2], b.cost, "Cost Map v3 (+Inflation)", "RdYlGn_r", vmin=1, vmax=20, label="비용")
    fig.suptitle(f"All Layers Final — {b.ts}", fontsize=14)
    fig.tight_layout()
    path = viz.savefig(fig, f"all_layers_final_{b.ts}.png")
    print(f"🖼️  {path}")


# ══════════════════════════════════════════════════════════
#  Virtual Rock Layer — 계획/리스크 단계 가상 큰돌 주입
# ══════════════════════════════════════════════════════════
def _rock_center_to_cell(rock: dict) -> tuple[int, int]:
    """rock 설정을 cell 좌표(row, col)로 변환한다."""
    if "row" in rock and "col" in rock:
        return int(round(float(rock["row"]))), int(round(float(rock["col"])))
    if "z" in rock and "x" in rock:
        return int(round(float(rock["z"]) / config.GRID_RES)), int(round(float(rock["x"]) / config.GRID_RES))
    raise ValueError(f"rock에는 row/col 또는 x/z가 필요합니다: {rock}")


def _circle_mask(shape: tuple[int, int], center_row: int, center_col: int, radius_cells: int) -> np.ndarray:
    rows, cols = shape
    rr, cc = np.ogrid[:rows, :cols]
    return (rr - center_row) ** 2 + (cc - center_col) ** 2 <= radius_cells ** 2


def build_virtual_rock_layers(shape: tuple[int, int], rocks: list[dict] | None = None):
    """
    config.VIRTUAL_ROCKS 기반 레이어 생성.

    반환:
      rock_mask     : 돌 본체. 통행 불가 + cost inf
      object_height : LoS 차폐용 추가 높이
      cover_mask    : 돌 주변 엄폐 영향권. open risk 감소
      soft_mask     : 운전상 조심해야 하는 주변부. cost 증가
    """
    rocks = rocks if rocks is not None else getattr(config, "VIRTUAL_ROCKS", [])
    rock_mask = np.zeros(shape, dtype=bool)
    object_height = np.zeros(shape, dtype=float)

    for rock in rocks:
        row, col = _rock_center_to_cell(rock)
        radius_m = float(rock.get("radius_m", config.GRID_RES))
        radius_cells = max(1, int(np.ceil(radius_m / config.GRID_RES)))

        # 맵 밖 좌표는 조용히 무시
        if row < 0 or col < 0 or row >= shape[0] or col >= shape[1]:
            print(f"[rocks] skip out-of-map rock: {rock}")
            continue

        mask = _circle_mask(shape, row, col, radius_cells)
        rock_mask |= mask
        object_height[mask] = np.maximum(object_height[mask], float(rock.get("height_m", 0.0)))

    inflate_cells = int(getattr(config, "ROCK_INFLATE_CELLS", 1))
    cover_cells = int(getattr(config, "ROCK_COVER_CELLS", 0))

    soft_mask = binary_dilation(rock_mask, iterations=inflate_cells) if inflate_cells > 0 else rock_mask.copy()
    cover_mask = binary_dilation(rock_mask, iterations=cover_cells) if cover_cells > 0 else rock_mask.copy()

    return rock_mask, object_height, cover_mask, soft_mask


def build_virtual_rock_masks(shape: tuple[int, int], rocks: list[dict] | None = None):
    """기존 코드 호환용. rock_mask, object_height만 반환."""
    rock_mask, object_height, _, _ = build_virtual_rock_layers(shape, rocks)
    return rock_mask, object_height


def apply_virtual_rocks(cost_map: np.ndarray,
                        obstacle_map: np.ndarray,
                        heightmap: np.ndarray | None = None,
                        rocks: list[dict] | None = None):
    """
    큰돌을 cost/obstacle에 반영한다.

    효과:
      1. 돌 본체: obstacle=True
      2. 돌 본체: cost=inf
      3. 돌 주변: soft cost 증가
      4. heightmap이 있으면 los_surface=heightmap+object_height 생성

    반환은 기존 호환을 위해 4개만 유지:
      new_cost, new_obstacle, rock_mask, los_surface
    """
    if not getattr(config, "ENABLE_VIRTUAL_ROCKS", False):
        return cost_map, obstacle_map, None, None

    new_cost = np.array(cost_map, dtype=float, copy=True)
    new_obstacle = np.array(obstacle_map, dtype=bool, copy=True)

    rock_mask, object_height, _cover_mask, soft_mask = build_virtual_rock_layers(new_cost.shape, rocks)
    if not rock_mask.any():
        return new_cost, new_obstacle, rock_mask, heightmap + object_height if heightmap is not None else None

    inflate_cost = float(getattr(config, "ROCK_INFLATE_COST", 10.0))

    # 본체는 통행 불가
    new_obstacle[rock_mask] = True
    new_cost[rock_mask] = np.inf

    # 주변은 hard block이 아니라 soft cost. 그래야 돌 근처를 완전히 봉쇄하지 않음.
    soft_ring = soft_mask & ~rock_mask & np.isfinite(new_cost)
    new_cost[soft_ring] += inflate_cost

    los_surface = heightmap + object_height if heightmap is not None else None
    print(f"[rocks] virtual rocks applied: {len(getattr(config, 'VIRTUAL_ROCKS', []))} rocks, {int(rock_mask.sum())} cells")
    return new_cost, new_obstacle, rock_mask, los_surface


def apply_virtual_rock_open_risk(open_risk: np.ndarray, shape: tuple[int, int] | None = None):
    """
    큰돌 주변 open risk 감소.

    개활지 한복판에 큰돌을 놓으면 그 주변은 엄폐/은폐 후보가 생긴 것이므로
    open risk가 그대로 1.0이면 실험 의미가 없다. 이 함수는 rock cover zone의
    open risk를 낮춰 combined cost와 risk_layer 그림에 반영한다.
    """
    if not getattr(config, "ENABLE_VIRTUAL_ROCKS", False):
        return open_risk, None, None

    shape = shape or open_risk.shape
    rock_mask, _object_height, cover_mask, _soft_mask = build_virtual_rock_layers(shape)
    if not rock_mask.any():
        return open_risk, rock_mask, cover_mask

    out = np.array(open_risk, dtype=float, copy=True)
    mult = float(getattr(config, "ROCK_OPEN_RISK_MULT", 0.25))
    body_val = float(getattr(config, "ROCK_BODY_OPEN_RISK", 0.0))

    cover_ring = cover_mask & ~rock_mask
    out[cover_ring] *= mult
    out[rock_mask] = body_val
    return out, rock_mask, cover_mask
