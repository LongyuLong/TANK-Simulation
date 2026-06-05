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
import numpy as np

from . import config, mapio, planning, risk

# 시뮬 실측 obstacle/terrain만 사용 — 합성 테스트 돌(VIRTUAL_ROCKS)은 끈다.
config.ENABLE_VIRTUAL_ROCKS = False

# patch 호환 기본 출발 (시뮬 /init 기준). 실제론 live position 사용 권장.
START_X, START_Z = 60.0, 27.23

# 모듈 known-set (SA 탐지가 갱신) — 글로벌은 stateless, 상태는 여기 보관
_known_enemies: list[risk.Enemy] = []


# ── 좌표 변환 ──────────────────────────────────────────────
def world_to_cell(x: float, z: float) -> tuple[int, int]:
    n = config.GRID_N
    r = int(round(z / config.GRID_RES))
    c = int(round(x / config.GRID_RES))
    return max(0, min(n - 1, r)), max(0, min(n - 1, c))


def cell_to_wp(r: int, c: int, y: float = 9.3) -> dict:
    return {"x": round(c * config.GRID_RES, 2), "y": y,
            "z": round(r * config.GRID_RES, 2)}


# 정적 지형물(맵 레이어로 베이크)만 — 동적(차/사람)은 베이크 X, 런타임 탐지로 처리.
STATIC_OBSTACLE_TYPES = ("rock", "wall", "house", "tree")


def _obs_type(o: dict) -> str:
    return str(o.get("type") or o.get("className") or o.get("name") or "").lower()


def is_static_obstacle(o: dict) -> bool:
    """정적 지형물(rock/wall/house/tree)인가. 차·사람 등은 False(런타임 인식)."""
    return any(s in _obs_type(o) for s in STATIC_OBSTACLE_TYPES)


def obstacles_to_cells(obstacles: list[dict], static_only: bool = True) -> list[tuple[int, int]]:
    """
    patch bbox 장애물(world) → 점유 셀.
      static_only=True : 정적 지형물만 (기본 — 동적 차/사람은 베이크 안 함).
      static_only=False: 전부 (런타임 일시 회피용, replan 시).
    """
    g, n = config.GRID_RES, config.GRID_N
    cells = set()
    for o in obstacles:
        if static_only and not is_static_obstacle(o):
            continue
        c0 = int(np.floor(o["x_min"] / g)); c1 = int(np.ceil(o["x_max"] / g))
        r0 = int(np.floor(o["z_min"] / g)); r1 = int(np.ceil(o["z_max"] / g))
        for r in range(max(0, r0), min(n, r1 + 1)):
            for c in range(max(0, c0), min(n, c1 + 1)):
                cells.add((r, c))
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
def plan_path(obstacles, sx=START_X, sz=START_Z, gx=None, gz=None,
              waypoint_file=None):
    """
    world 장애물+목표 → 우리 plan_threat(Option S) → world waypoints.
    반환: [{'x','y','z'}, ...] (patch pursuit 가 그대로 소비) 또는 None.
    """
    if gx is None or gz is None:
        return None
    start = world_to_cell(sx, sz)
    goal  = world_to_cell(gx, gz)
    extra = obstacles_to_cells(obstacles or [])

    path, st = planning.plan_threat(
        start, goal, _known_enemies,
        extra_obstacles=extra, save=False, plot=False, verbose=True)
    if path is None:
        return None
    return [cell_to_wp(r, c) for (r, c) in path]


# ── SA: 런타임 접촉 재계획 (app.py 가 탐지 시 호출) ─────────
def replan(current_xz, goal_xz, obstacles=None):
    """
    현재 world 위치→목표, 갱신된 known-set/장애물로 재계획 → world waypoints.
      obstacles : 이 replan에서 피할 장애물 (정적 지형 + 런타임 탐지한 일시 동적).
                  caller(app.py)가 큐레이션해 전달 → 여기선 전부 베이크(static_only=False).
    """
    extra = obstacles_to_cells(obstacles or [], static_only=False)
    path, st = planning.replan_on_contact(
        world_to_cell(*current_xz), world_to_cell(*goal_xz), _known_enemies,
        extra_obstacles=extra)
    return [cell_to_wp(r, c) for (r, c) in path] if path else None


def assess(current_xz, goal_xz, enemy_xz, etype="tank"):
    """탐지 적에 대한 교전 결심 근거 (engage/bypass 판단용)."""
    new_enemy = risk.Enemy(etype, [world_to_cell(*enemy_xz)])
    a = planning.assess_contact(world_to_cell(*current_xz),
                                world_to_cell(*goal_xz),
                                _known_enemies, new_enemy)
    bp = a.pop("bypass_path", None)
    a["bypass_waypoints"] = [cell_to_wp(r, c) for (r, c) in bp] if bp else None
    return a


def decide(assessment, policy="balanced"):
    """교전 결심 'engage'|'bypass' (참고 stub — 실제 소유는 로컬/web 운용자)."""
    return planning.decide_engagement(assessment, policy=policy)
