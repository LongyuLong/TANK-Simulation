"""
[채점기] 전역 경로 평가 — 탐지→피격 체인 + 기동시간   (ATP 3-20.15 OAKOC)

비용맵(ridge/open/los_risk 가중합)과 **레벨이 다른** ground-truth 목적함수.
  - 탐지는 비용맵의 los_risk 대신 risk.is_visible() 로 직접 판정 (순환논리 회피).
  - 경로의 "좋음" = P(생존) 과 기동시간 T 의 임무별 가중합(낮을수록 좋음).

체인:
  P(피격) = P(탐지)·P(교전|탐지)·P(명중|교전)·P(살상|명중)
  P(생존) = Π_seg [ 1 − P(피격_seg) ]

적 모델(risk.Enemy): etype + positions[].  patrol 은 경로 셀들을
**정적 관측 집합**으로 본다(= 적 순찰로를 사전 정보로 안다는 가정, 시간 비동기).

precompute_threat() 로 적-의존 항(거리·교전·명중)을 셀별 1회 계산해두고,
score_path() 는 셀-의존 항(은폐·엄폐·노출시간)만 곱해 빠르게 채점 → 튜닝 루프용.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from . import config, mapio, risk


# ══════════════════════════════════════════════════════════
#  거리-확률 곡선
# ══════════════════════════════════════════════════════════
def p_range(d_m: float, detect_range_m: float) -> float:
    """탐지 거리감쇠 (선형). 멀수록 탐지 어려움."""
    return max(0.0, 1.0 - d_m / detect_range_m)


def p_engage(d_m: float, weapon_range_m: float) -> float:
    """교전 결심 — 2/3 규칙(¶4-156). 2/3선 이내 최대, MEL 초과 0."""
    two_thirds = weapon_range_m * (2.0 / 3.0)
    if d_m <= two_thirds:
        return 1.0
    if d_m <= weapon_range_m:
        return 1.0 - (d_m - two_thirds) / (weapon_range_m - two_thirds)
    return 0.0


def p_hit(d_m: float, etype: str) -> float:
    """명중확률 — 거리 지수감쇠 (무기별)."""
    ph_max, k = config.WEAPON_PH[etype]
    return ph_max * np.exp(-k * d_m)


# ══════════════════════════════════════════════════════════
#  통행성 → 속도
# ══════════════════════════════════════════════════════════
def mobility_tier(slope_deg: float, blocked: bool) -> str:
    if blocked:
        return "blocked"
    lo, hi = config.SLOPE_TIER
    if slope_deg < lo:
        return "unrestricted"
    if slope_deg < hi:
        return "restricted"
    return "severely_restricted"


# ══════════════════════════════════════════════════════════
#  Precompute — 적-의존 위협장 (셀별, 경로 무관)
# ══════════════════════════════════════════════════════════
@dataclass
class Threat:
    intensity: np.ndarray   # max_적 (p_range·p_engage·p_hit), 가시·교전 가능시
    concealment: np.ndarray # 은폐 → P(탐지) 배율 (0.1~1)
    cover: np.ndarray       # 엄폐 → P(살상) 배율 (COVER_FLOOR~1)



def _safe_score_result(P_survive: float = 0.0,
                       time_s: float = float("inf"),
                       t_norm: float = float("inf"),
                       score: float = float("inf"),
                       n_blocked: int = 0,
                       per_seg: list | None = None,
                       reason: str = "") -> dict:
    """
    runlog.py가 항상 기대하는 키를 보장하는 안전 반환 함수.
    경로가 막히거나 비정상이어도 KeyError가 나지 않게 한다.
    """
    return {
        "P_survive": P_survive,
        "time_s": time_s,
        "t_norm": t_norm,
        "score": score,
        "n_blocked": n_blocked,
        "per_seg": per_seg or [],
        "reason": reason,
    }


def precompute_threat(enemies: list[risk.Enemy], bundle: mapio.MapBundle,
                      hm_los: np.ndarray | None = None) -> Threat:
    """
    적 위치 고정 가정 하에 셀별 위협을 1회 계산.
      intensity[cell] = max over (적위치) [ p_range·p_engage·p_hit ]
                        단, is_visible 이고 무기/탐지 사거리 이내일 때만.
    은폐·엄폐는 적과 무관한 지형항이라 별도 배열로.

    hm_los : LoS(is_visible) 계산용 차폐면. None이면 지면 heightmap.
             돌이 있으면 los_surface(=지형+돌높이)를 넘겨 돌 그림자(사각)를 반영(P4).
    """
    hm = bundle.heightmap_filled
    los = hm_los if hm_los is not None else hm
    rows, cols = hm.shape
    intensity = np.zeros((rows, cols), dtype=float)

    for e in enemies:
        det   = config.DETECT_RANGE[e.etype]
        obs_h = config.OBS_HEIGHT[e.etype]
        wrng  = config.WEAPON_RANGE[e.etype]
        reach = min(det, wrng)                 # 보지도 못하는 거리에선 교전 불가
        det_cells = int(reach / config.GRID_RES)
        for (er, ec) in e.positions:
            r0, r1 = max(0, er - det_cells), min(rows, er + det_cells + 1)
            c0, c1 = max(0, ec - det_cells), min(cols, ec + det_cells + 1)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    d = np.hypot(r - er, c - ec) * config.GRID_RES
                    if d > reach:
                        continue
                    if not risk.is_visible(los, er, ec, obs_h, r, c, config.K2_HEIGHT_M):
                        continue
                    val = p_range(d, det) * p_engage(d, wrng) * p_hit(d, e.etype)
                    if val > intensity[r, c]:
                        intensity[r, c] = val

    # 지형항 (적 무관) — fresh 계산 (비용맵 저장본을 끌어오지 않음)
    ridge = risk.make_ridge_risk(hm)
    open_ = risk.make_open_risk(bundle.obstacle, bundle.cost)
    concealment = np.clip(1.0 - 0.4 * ridge - 0.4 * open_, 0.1, 1.0)
    cover = config.COVER_FLOOR + (1.0 - config.COVER_FLOOR) * open_
    return Threat(intensity, concealment, cover)


# ══════════════════════════════════════════════════════════
#  채점
# ══════════════════════════════════════════════════════════
def score_path(path: list[tuple[int, int]],
               enemies: list[risk.Enemy],
               bundle: mapio.MapBundle,
               mission: str = "attack",
               precomp: Threat | None = None,
               ablate: frozenset = frozenset()) -> dict:
    """
    path    : [(row,col), ...]  A* 출력 셀
    enemies : list[risk.Enemy]
    bundle  : MapBundle
    ablate  : 중립화(=1.0)할 보호 항 집합. {"concealment","cover","exposure"}.
              민감도 분석용 — 해당 항의 방호효과를 끈다.
    반환    : {P_survive, time_s, t_norm, score, n_blocked, per_seg, reason}
              score 는 비용(낮을수록 좋음).
              주의: 경로가 막혀도 runlog.py에서 KeyError가 나지 않도록
              t_norm 포함 모든 기본 키를 항상 반환한다.
    """
    if precomp is None:
        precomp = precompute_threat(enemies, bundle)

    # path가 없거나 너무 짧으면 채점 불가.
    # 이 경우에도 runlog.py가 기대하는 키는 모두 반환한다.
    if not path or len(path) < 2:
        return _safe_score_result(
            P_survive=0.0,
            time_s=0.0,
            t_norm=0.0,
            score=float("inf"),
            n_blocked=1,
            per_seg=[],
            reason="empty_or_too_short_path",
        )

    T = precomp
    slope, obst = bundle.slope, bundle.obstacle
    rows, cols = slope.shape
    ab_con = "concealment" in ablate
    ab_cov = "cover" in ablate
    ab_exp = "exposure" in ablate

    log_survive = 0.0
    total_time  = 0.0
    per_seg     = []

    for i in range(len(path) - 1):
        r, c   = path[i]
        nr, nc = path[i + 1]

        # 경로 셀이 맵 밖으로 나가면 invalid 처리.
        if not (0 <= r < rows and 0 <= c < cols and 0 <= nr < rows and 0 <= nc < cols):
            per_seg.append({
                "cell": (r, c),
                "next_cell": (nr, nc),
                "tier": "out_of_bounds",
                "dt": float("inf"),
                "intensity": 0.0,
                "concealment": 1.0,
                "cover": 1.0,
                "exposure": 1.0,
                "p_hit": 1.0,
                "blocked": True,
                "reason": "out_of_bounds",
            })
            return _safe_score_result(
                P_survive=0.0,
                time_s=float("inf"),
                t_norm=float("inf"),
                score=float("inf"),
                n_blocked=1,
                per_seg=per_seg,
                reason="out_of_bounds",
            )

        seg_len = np.hypot(nr - r, nc - c) * config.GRID_RES   # 대각=√2·res

        tier = mobility_tier(float(slope[r, c]), bool(obst[r, c]))
        v = config.TIER_SPEED[tier]
        if v == 0.0:
            per_seg.append({
                "cell": (r, c),
                "next_cell": (nr, nc),
                "tier": tier,
                "dt": float("inf"),
                "intensity": float(T.intensity[r, c]),
                "concealment": float(T.concealment[r, c]),
                "cover": float(T.cover[r, c]),
                "exposure": 1.0,
                "p_hit": 1.0,
                "blocked": True,
                "reason": "blocked_cell",
            })
            return _safe_score_result(
                P_survive=0.0,
                time_s=float("inf"),
                t_norm=float("inf"),
                score=float("inf"),
                n_blocked=1,
                per_seg=per_seg,
                reason="blocked_cell",
            )
        dt = seg_len / v
        total_time += dt

        exposure = 1.0 if ab_exp else 1.0 - np.exp(-dt / config.T_EXPOSE)
        conceal  = 1.0 if ab_con else T.concealment[r, c]
        cover    = 1.0 if ab_cov else T.cover[r, c]
        p_hit_seg = (T.intensity[r, c] * conceal * exposure
                     * config.PK_BASE * cover)
        p_hit_seg = min(p_hit_seg, 0.99)
        log_survive += np.log(max(1e-9, 1.0 - p_hit_seg))
        per_seg.append({
            "cell": (r, c), "tier": tier, "dt": dt,
            "intensity"  : float(T.intensity[r, c]),
            "concealment": float(conceal),
            "cover"      : float(cover),
            "exposure"   : float(exposure),
            "p_hit"      : float(p_hit_seg),
        })

    P_survive = float(np.exp(log_survive))

    # 시간 정규화 기준 = 직선거리 / 최대속도
    straight = np.hypot(path[-1][0] - path[0][0],
                        path[-1][1] - path[0][1]) * config.GRID_RES
    t_ref  = straight / max(config.TIER_SPEED.values())
    t_norm = total_time / t_ref if t_ref > 0 else 0.0

    w_surv, w_time = config.MISSION_WEIGHTS[mission]
    score = w_surv * (1.0 - P_survive) + w_time * t_norm

    return {
        "P_survive": P_survive,
        "time_s"   : total_time,
        "t_norm"   : t_norm,
        "score"    : score,
        "n_blocked": 0,
        "per_seg"  : per_seg,
    }


# ══════════════════════════════════════════════════════════
#  위협-인지 비용맵 (채점기 → 계획기 연결, Option S)
# ══════════════════════════════════════════════════════════
def hazard_field(bundle: mapio.MapBundle, precomp: Threat) -> np.ndarray:
    """
    셀별 hazard = −log(1 − p_hit_cell), 단위 셀 1회 체류 기준.
    Σ hazard = −log(P_survive) 이므로 A* 비용에 더하면 생존 최대화와 등가.
    """
    slope, obst = bundle.slope, bundle.obstacle
    rows, cols = precomp.intensity.shape
    haz = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            tier = mobility_tier(float(slope[r, c]), bool(obst[r, c]))
            v = config.TIER_SPEED[tier]
            if v == 0.0:
                continue
            dt = config.GRID_RES / v
            exposure = 1.0 - np.exp(-dt / config.T_EXPOSE)
            ph = (precomp.intensity[r, c] * precomp.concealment[r, c] * exposure
                  * config.PK_BASE * precomp.cover[r, c])
            ph = min(ph, 0.99)
            haz[r, c] = -np.log(max(1e-9, 1.0 - ph))
    return haz


def threat_cost(bundle: mapio.MapBundle, precomp: Threat,
                w_surv: float | None = None,
                base_cost: np.ndarray | None = None) -> np.ndarray:
    """
    Option S 플래너 입력: base 지형 비용 + W_SURV_PLAN·hazard. inf(통과불가) 보존.
      base_cost : None이면 bundle.cost. 돌 적용 비용(apply_virtual_rocks 결과)을
                  넘기면 돌 inf+마진까지 반영된다.
    """
    if w_surv is None:
        w_surv = config.W_SURV_PLAN
    base = bundle.cost if base_cost is None else base_cost
    cost = base + w_surv * hazard_field(bundle, precomp)
    cost[np.isinf(base)] = np.inf
    return cost


# ══════════════════════════════════════════════════════════
#  유틸 — 저장된 path_*.npy(미터) → 셀 리스트
# ══════════════════════════════════════════════════════════
def load_path_cells(npy_path) -> list[tuple[int, int]]:
    arr = np.load(npy_path)   # [[r*res, c*res], ...]
    g = config.GRID_RES
    return [(int(round(r / g)), int(round(c / g))) for r, c in arr]
