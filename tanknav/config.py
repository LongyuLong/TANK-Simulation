"""
tanknav 전역 설정 — 단일 소스 (Single Source of Truth)

이전에는 GRID_RES, K2 제원, 가중치, 타임스탬프가 파일마다 흩어져
하드코딩되어 있었음. 모든 설정을 여기로 모음.
"""
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
# 데이터(.npy/.png)는 프로젝트 루트(fin/)에 저장됨
DATA_DIR = Path(__file__).resolve().parent.parent

# ── 격자 / 맵 ─────────────────────────────────────────────
MAP_SIZE = 300          # 맵 한 변 (m)
GRID_RES = 5.0          # 셀 하나 = 5m
GRID_N   = int(MAP_SIZE // GRID_RES) + 1   # 61

# ── K2 전차 제원 ──────────────────────────────────────────
K2_HEIGHT_M = 2.4       # 전고 — 능선 노출 / 열상센서 높이 기준
K2_WIDTH_M  = 3.6       # 전폭 — 엄폐 판단 참고

# ── Cost 생성 (mapping) ───────────────────────────────────
# 경사도(°) 구간별 비용
SLOPE_COST = [
    (10, 2.0),          # >10° → 2.0
    (20, 5.0),          # >20° → 5.0
    (30, float("inf")), # >30° → 통과 불가
]
CLIFF_DIFF_M    = 5.0   # 3x3 내 고도차 5m 이상 = 절벽 (inf)
INFLATE_ITERS   = 2     # 장애물 팽창 반복
INFLATE_COST    = 8.0   # 팽창 영역 비용

# ── 리스크 레이어 ─────────────────────────────────────────
RIDGE_RADIUS_CELLS = 4              # 능선 탐색 반경 (20m)
RIDGE_EXPOSE_THR   = K2_HEIGHT_M    # 주변보다 2.4m↑ = 노출

OPEN_RADIUS_CELLS = 6               # 개활지 엄폐밀도 반경 (30m)
OPEN_MIN_COVER    = 0.05            # 엄폐 5% 미만 = 완전 개활지
OPEN_MAX_COVER    = 0.20            # 20% 이상이면 리스크 0

# 적 탐지 범위 (m) / 관측 높이 (지면 위 m)
#   시뮬레이터 실제 제원: LiDAR 최대 탐지거리 80m (센서가 병목).
#   탐지가 무기 사거리(130m)보다 짧아 → 탐지된 표적은 항상 교전 가능.
DETECT_RANGE = {"infantry": 80,  "tank": 80,  "patrol": 80}
OBS_HEIGHT   = {"infantry": 1.8, "tank": 2.4, "patrol": 2.4}

# ── 통합 코스트 가중치 ────────────────────────────────────
W_BASE  = 1.0
W_RIDGE = 3.0   # 능선 피탐지 — 치명적
W_OPEN  = 2.0   # 개활지 노출
W_LOS   = 5.0   # 적 직접 가시선 — 가장 치명적

# ── A* 계획 ───────────────────────────────────────────────
HEIGHT_PENALTY = 0.1   # 이동 비용에 더하는 고도차 가중치

# Theta* 스무싱 — 직선 단축 수용 허용오차.
#   단축 세그먼트 비용 적분 ≤ 원본 구간 비용 × (1+TOL) 일 때만 채택.
#   0.0 = 위험 증가 절대 금지(안전한 경로 내 스무싱만). 키우면 더 매끈하되 위험↑ 허용.
THETA_COST_TOL = 0.0

# ── [설계 메모 / 미구현] 임무 시급성(urgency) 단일 노브 ───────────────
#   "노출 감수하고 빠르게" ↔ "느려도 안전"을 하나의 파라미터로 묶는 구상.
#   urgency ∈ [0,1] 하나가 두 레벨을 동시에 제어:
#     ① 계획 레벨(주): 어느 회랑 — 안전 우회 vs 지름길.
#        예) W_SURV_PLAN = lerp(높음, 낮음, urgency)   ← 채점기 직결(threat_cost) 채택 시
#            또는 휴리스틱이면 W_LOS/W_RIDGE 를 거리비용 대비 낮춤.
#     ② 스무싱 레벨(부): 그 회랑 안에서 코너 컷 정도.
#        예) THETA_COST_TOL = lerp(0.0, 0.3, urgency)
#   주의: THETA_COST_TOL 단독은 "이미 안전한 경로의 코너 컷"만 가능(회랑 선택 불가).
#   → 진짜 시급성은 계획-레벨 노브가 핵심. A1(플래너 cost 방식) 결정 후 배선 예정.
#   현재는 동작에 영향 없음(주석 전용).
# URGENCY = 0.0
# def _urgency_knobs(u):
#     return dict(W_SURV_PLAN=lerp(HIGH, LOW, u), THETA_COST_TOL=0.3*u)

# ── 채점기 (eval) ─────────────────────────────────────────
# 통행성 3단계 → 속도 (m/s).  ¶2-54~56
#   slope < SLOPE_TIER[0]            → unrestricted
#   SLOPE_TIER[0] ≤ slope < [1]      → restricted
#   slope ≥ SLOPE_TIER[1]            → severely_restricted
SLOPE_TIER = (10.0, 20.0)
TIER_SPEED = {
    "unrestricted"       : 15.0,   # ~54 km/h
    "restricted"         : 6.0,    # ~22 km/h
    "severely_restricted": 2.0,    # ~7 km/h
    "blocked"            : 0.0,    # 통과 불가
}

T_EXPOSE = 5.0    # 노출시간 포화상수 (s) — dt 동안 탐지확률 1-exp(-dt/T_EXPOSE)

# 피격 체인 파라미터
PK_BASE     = 0.7    # 명중 시 기본 살상확률
COVER_FLOOR = 0.6    # 완전 엄폐(open_risk=0) 시 cover_factor 하한 (Pk 40%↓)

# 위협-인지 A* 비용 가중 (Option S): threat_cost = base + W_SURV_PLAN·hazard
#   hazard = −log(1−p_hit_cell). Σhazard = −log(P_survive) → A* 최소화 = 생존 최대화.
#   클수록 생존 우선(우회↑), 작을수록 지형/거리 우선(직선↑).
W_SURV_PLAN = 20.0

# ── 맵 설치 오브젝트 타입별 파라미터 ─────────────────────────
# height_mult : K2 전고(K2_HEIGHT_M=2.4m) 대비 높이 배수.
#               los_surface(능선·개활·은폐·LoS 위협)에만 반영 — heightmap/slope 에는 반영 X.
# risk        : True면 위 risk 레이어 반영. cost(inf+마진)는 활성 타입 모두 적용.
# 맵 prefab(Rock002 등)은 정규화(소문자+앞자리0제거: Rock002→rock2)되어 이 키와 매칭.
# 주석 해제 = 해당 타입을 계획/리스크에 포함.
OBJECT_TYPES = {
    # "rock1": {"height_mult": 1.1, "risk": True},
    "rock2": {"height_mult": 1.1, "risk": True},     # ← Test1.map 설치물(현재 활성)
    # "car1":  {"height_mult": 0.8, "risk": True},
    # "car2":  {"height_mult": 0.8, "risk": True},
    # "car3":  {"height_mult": 0.8, "risk": True},
    # "wall1": {"height_mult": 1.5, "risk": True},
    # "wall2": {"height_mult": 1.5, "risk": True},
}

# 적 무기 유효사거리 (m) — etype 키.  평지-평지 최대사거리 130m (sim 제원).
#   DETECT(80) < WEAPON(130) 이라 2/3선(~87m)이 탐지밖 → p_engage 사실상 1.
#   (고지에서 사거리 증가 규칙은 v2)
WEAPON_RANGE = {"infantry": 130, "tank": 130, "patrol": 130}
# (Ph_max, 거리 감쇠율 /m) — 거리-명중률 지수곡선
WEAPON_PH = {
    "infantry": (0.70, 0.0020),
    "tank"    : (0.95, 0.0008),
    "patrol"  : (0.95, 0.0008),
}

# 임무별 (생존 가중, 시간 가중)
#   survival = 시간 비활성(순수 생존). 단 계획 단계에선 경로길이 정규화 재도입 필요.
MISSION_WEIGHTS = {
    "survival": (1.0, 0.0),
    "attack"  : (0.4, 0.6),
    "defend"  : (0.8, 0.2),
    "recon"   : (0.9, 0.1),
}

# ── 가상 큰돌 실험 레이어 ─────────────────────────────────────
# 목적: Yakis.py로 딴 원본 맵은 그대로 두고, 계획 단계에서만 큰돌을 추가해
#       A*가 돌을 피해 재계획하는지 확인한다.
# 좌표는 기본적으로 cell 좌표(row=Z, col=X)를 권장한다.
# world 좌표를 쓰려면 {"x": 100, "z": 150, ...} 형태도 가능하다.
ENABLE_VIRTUAL_ROCKS = True

# 예시 돌 위치. 네가 원하는 위치로 row/col만 바꿔가며 실험하면 된다.
# radius_m: 돌 반경, height_m: 나중에 LoS/엄폐 실험용 높이
VIRTUAL_ROCKS = [
    # 1. 중앙 개활지 실험용
    # 주변에 기존 장애물이 적어서 open risk 감소가 가장 잘 보이는 위치
    {"row": 25, "col": 30, "radius_m": 7.0, "height_m": 2.5, "name": "open_field_center"},

    # 2. 우측 개활지 실험용
    # 경로가 중앙으로 갈지, 돌 주변 엄폐 구역을 활용해 우측으로 갈지 비교 가능
    {"row": 33, "col": 41, "radius_m": 7.0, "height_m": 2.5, "name": "open_field_right"},

    # 3. 기존 이동 경로 우회 확인용
    # A* 경로가 자주 지나가는 중후반 길목에 배치해서 우회 여부 확인
    {"row": 39, "col": 36, "radius_m": 8.0, "height_m": 2.5, "name": "route_detour_test"},

    # 4. 목적지 접근 전 엄폐물
    # goal에 너무 붙지 않으면서 마지막 접근 경로가 달라지는지 확인
    {"row": 47, "col": 46, "radius_m": 7.0, "height_m": 2.5, "name": "goal_approach_cover_1"},

    # 5. 목적지 근처 최종 엄폐물
    # goal(55,55)을 직접 막지 않고, 최종 접근 방향 선택에 영향을 주는 위치
    {"row": 51, "col": 50, "radius_m": 6.0, "height_m": 2.5, "name": "goal_approach_cover_2"},
]
# 돌 본체 주변 비용 증가. 본체는 inf, 주변은 soft cost.
ROCK_INFLATE_CELLS = 1
ROCK_INFLATE_COST = 10.0

# 돌 주변 엄폐 효과.
# rock_mask = 돌 본체(통행불가), cover_mask = 돌 주변 엄폐 가능 구역.
# cover_mask 안의 open risk를 낮춰서 "개활지에 돌을 놓으면 엄폐가 생긴다"는 효과를 반영한다.
ROCK_COVER_CELLS = 1            # 2셀 = 약 10m 주변을 엄폐 영향권으로 봄
ROCK_OPEN_RISK_MULT = 0.25    # 돌 주변 open risk를 25%로 감소
ROCK_BODY_OPEN_RISK = 0.0     # 돌 본체는 장애물이므로 개활지 risk 표시에서는 0 처리
ROCK_USE_LOS_SURFACE = True   # LoS 계산 시 heightmap + object_height 사용

# PNG에서 돌 위치를 직접 보이게 표시할지 여부
SHOW_VIRTUAL_ROCKS_ON_PLOT = True
