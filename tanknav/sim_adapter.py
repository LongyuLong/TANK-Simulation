"""
[시뮬레이터 어댑터] patch(web+통신+추종) ↔ 글로벌 플래너(Option S)

patch의 path_planner(평면 A*)를 우리 plan_threat로 교체하기 위한 변환 계층.
  - 좌표: world(x,z, m) ↔ 우리 셀 (row=z/GRID_RES, col=x/GRID_RES). 0~300, 5m.
           (risk.parse_enemies 와 동일 규약 — 축 뒤집힘 없음)
  - 지형: 우리 heightmap/cost(MapBundle) 그대로.
  - 장애물: patch가 주는 world bbox(new_map3 + 런타임 탐지) → 셀 주입(inf+마진).
  - 적: 모듈 known-set(탐지로 갱신) → threat_cost.

patch app.py 교체:
    from path_planner import plan_path, START_X, START_Z
  →  from tanknav.sim_adapter import plan_path, START_X, START_Z
(나머지 app.py / pursuit / dashboard / 시뮬 엔드포인트는 그대로)
"""
from __future__ import annotations
import re
import numpy as np

from . import config, mapio, planning, risk

# 시뮬 실측 obstacle/terrain만 사용 — 합성 테스트 돌(VIRTUAL_ROCKS)은 끈다.
config.ENABLE_VIRTUAL_ROCKS = False

# patch 호환 기본 출발 (시뮬 /init 기준). 실제론 live position 사용 권장.
START_X, START_Z = 60.0, 27.23

# 추종기용 웨이포인트 간격(m). Theta* 희소 waypoint를 이 간격으로 보간 → 추종 정확도↑.
WAYPOINT_SPACING_M = 10.0

# 모듈 known-set (SA 탐지가 갱신) — 글로벌은 stateless, 상태는 여기 보관
_known_enemies: list[risk.Enemy] = []

# 맵 설치 정적 오브젝트 (app.py가 맵 로드 시 set_static_obstacles로 1회 등록)
_static_objects: list[dict] = []


# ── 좌표 변환 ──────────────────────────────────────────────
def world_to_cell(x: float, z: float) -> tuple[int, int]:
    n = config.GRID_N
    r = int(round(z / config.GRID_RES))
    c = int(round(x / config.GRID_RES))
    return max(0, min(n - 1, r)), max(0, min(n - 1, c))


def cell_to_wp(r: float, c: float, y: float = 9.3) -> dict:
    return {"x": round(c * config.GRID_RES, 2), "y": y,
            "z": round(r * config.GRID_RES, 2)}


def densify_cells(path, spacing_m: float = WAYPOINT_SPACING_M):
    """희소 Theta* 셀 경로 → spacing_m 간격으로 보간(추종 정확도↑)."""
    if not path or len(path) < 2:
        return path
    g = config.GRID_RES
    out = []
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        seg_m = np.hypot(r1 - r0, c1 - c0) * g
        n = max(1, int(seg_m // spacing_m))
        for i in range(n):
            t = i / n
            out.append((r0 + (r1 - r0) * t, c0 + (c1 - c0) * t))
    out.append((float(path[-1][0]), float(path[-1][1])))
    return out


def _cells_to_waypoints(path):
    return [cell_to_wp(r, c) for (r, c) in densify_cells(path)]


# ── 오브젝트 타입 분류 (config.OBJECT_TYPES 기반) ──────────
def _type_key(o: dict) -> str:
    """prefab/타입명 → config 키. 예: 'Rock002_5' → 'rock2'."""
    name = str(o.get("type") or o.get("name") or o.get("className") or "")
    base = name.split("_")[0]
    m = re.match(r"([A-Za-z]+)0*(\d+)", base)
    return (m.group(1).lower() + m.group(2)) if m else base.lower()


def _bbox_cells(o: dict) -> list[tuple[int, int]]:
    """world bbox → 점유 셀."""
    g, n = config.GRID_RES, config.GRID_N
    c0 = int(np.floor(o["x_min"] / g)); c1 = int(np.ceil(o["x_max"] / g))
    r0 = int(np.floor(o["z_min"] / g)); r1 = int(np.ceil(o["z_max"] / g))
    return [(r, c) for r in range(max(0, r0), min(n, r1 + 1))
            for c in range(max(0, c0), min(n, c1 + 1))]


def set_static_obstacles(obstacles) -> None:
    """맵 로드 시 정적 오브젝트 등록 (app.py가 1회 호출). 계획/리스크/레이어 공통 사용."""
    global _static_objects
    _static_objects = list(obstacles or [])


def _build_object_grids(bundle):
    """
    _static_objects + config.OBJECT_TYPES → (cost_cells, los_surface, obstacle_aug).
      활성(주석해제) 타입만 반영. risk=True 면 los_surface(높이)·obstacle_aug에 추가.
      비활성/미등록 타입은 무시. heightmap/slope 는 건드리지 않음.
    """
    hm = np.asarray(bundle.heightmap_filled, dtype=float)
    los = hm.copy()
    obst = np.asarray(bundle.obstacle, dtype=bool).copy()
    cost_cells: list[tuple[int, int]] = []
    for o in _static_objects:
        spec = config.OBJECT_TYPES.get(_type_key(o))
        if spec is None:
            continue                       # 비활성/미등록 타입 → 무시
        h = float(spec.get("height_mult", 1.0)) * config.K2_HEIGHT_M
        for (r, c) in _bbox_cells(o):
            cost_cells.append((r, c))      # 통행불가(cost)는 활성 타입 모두
            if spec.get("risk", True):     # risk 반영: 높이(los) + 엄폐(obstacle)
                los[r, c] = hm[r, c] + h
                obst[r, c] = True
    return cost_cells, los, obst


def _transient_cells(obstacles) -> list[tuple[int, int]]:
    """런타임 일시 장애물(차/사람 등) → cost 셀 (risk 미반영)."""
    cells = set()
    for o in obstacles or []:
        cells.update(_bbox_cells(o))
    return list(cells)


# ── known-set 관리 (SA 탐지가 호출) ───────────────────────
def set_enemies(enemies: list[risk.Enemy]) -> None:
    global _known_enemies
    _known_enemies = list(enemies)


def add_enemy(x: float, z: float, etype: str = "tank") -> None:
    _known_enemies.append(risk.Enemy(etype, [world_to_cell(x, z)]))


def clear_enemies() -> None:
    _known_enemies.clear()


# ── patch 호환 진입점 (path_planner.plan_path 대체) ─────────
def plan_path(obstacles=None, sx=START_X, sz=START_Z, gx=None, gz=None,
              waypoint_file=None):
    """
    목표(world) → 우리 plan_threat(Option S) → world waypoints.
      정적 오브젝트: set_static_obstacles 등록분(config 활성 타입) → cost+risk.
      obstacles(인자): 런타임 일시 장애물(동적) → cost 회피만.
    반환: [{'x','y','z'}, ...] (patch pursuit 소비) 또는 None.
    """
    if gx is None or gz is None:
        return None
    bundle = mapio.load_maps()
    cost_cells, los, obst = _build_object_grids(bundle)
    cost_cells = cost_cells + _transient_cells(obstacles)

    path, st = planning.plan_threat(
        world_to_cell(sx, sz), world_to_cell(gx, gz), _known_enemies,
        extra_obstacles=cost_cells, los_surface=los, obstacle_override=obst,
        save=False, plot=False, verbose=True)
    return _cells_to_waypoints(path) if path else None


# ── SA: 런타임 접촉 재계획 (app.py 가 탐지 시 호출) ─────────
def replan(current_xz, goal_xz, obstacles=None):
    """
    현재 world 위치→목표 재계획 → world waypoints.
      정적 오브젝트 risk + obstacles(런타임 일시) cost회피 + 갱신된 known-set 반영.
    """
    bundle = mapio.load_maps()
    cost_cells, los, obst = _build_object_grids(bundle)
    cost_cells = cost_cells + _transient_cells(obstacles)
    path, st = planning.replan_on_contact(
        world_to_cell(*current_xz), world_to_cell(*goal_xz), _known_enemies,
        extra_obstacles=cost_cells, los_surface=los, obstacle_override=obst)
    return _cells_to_waypoints(path) if path else None


def assess(current_xz, goal_xz, enemy_xz, etype="tank"):
    """탐지 적에 대한 교전 결심 근거 (engage/bypass 판단용)."""
    bundle = mapio.load_maps()
    _, los, obst = _build_object_grids(bundle)
    new_enemy = risk.Enemy(etype, [world_to_cell(*enemy_xz)])
    a = planning.assess_contact(world_to_cell(*current_xz),
                                world_to_cell(*goal_xz),
                                _known_enemies, new_enemy,
                                los_surface=los, obstacle_override=obst)
    bp = a.pop("bypass_path", None)
    a["bypass_waypoints"] = _cells_to_waypoints(bp) if bp else None
    return a


def decide(assessment, policy="balanced"):
    """교전 결심 'engage'|'bypass' (참고 stub — 실제 소유는 로컬/web 운용자)."""
    return planning.decide_engagement(assessment, policy=policy)


# ── 대시보드 레이어 데이터 (JSON) ──────────────────────────
def layers():
    """
    대시보드 표시용 레이어 그리드(61×61) 묶음.
      정적: heightmap/slope/ridge/open   동적(known-set 의존): threat/concealment/cover
    각 grid는 [row=z][col=x]. JS가 min-max 정규화해 색칠.
    """
    from . import eval as _eval
    bundle = mapio.load_maps()
    hm = bundle.heightmap_filled
    _, los, obst = _build_object_grids(bundle)   # 오브젝트(config) 반영
    ridge = risk.make_ridge_risk(los)            # 능선: los_surface
    open_ = risk.make_open_risk(obst, bundle.cost)  # 개활: 오브젝트 포함 마스크
    pc = _eval.precompute_threat(_known_enemies, bundle,
                                 hm_los=los, obstacle_override=obst)

    def g(a):
        return np.asarray(a, dtype=float).round(3).tolist()

    return {
        "grid_n": config.GRID_N,
        "grid_res": config.GRID_RES,
        "enemies": [[int(r), int(c)] for e in _known_enemies for (r, c) in e.positions],
        "objects": [[int(r), int(c)] for (r, c) in
                    {cell for o in _static_objects if config.OBJECT_TYPES.get(_type_key(o))
                     for cell in _bbox_cells(o)}],
        "grids": {
            "heightmap":   g(hm),
            "slope":       g(bundle.slope),
            "ridge_risk":  g(ridge),
            "open_risk":   g(open_),
            "threat":      g(pc.intensity),
            "concealment": g(pc.concealment),
            "cover":       g(pc.cover),
        },
    }
