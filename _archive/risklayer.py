import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter
from datetime import datetime

# ══════════════════════════════════════════════════════════
#  K2 전차 기반 리스크 레이어 v1
#  - Ridge Risk  : 능선 노출 (피탐지)
#  - Open Risk   : 개활지 노출 (엄폐 부족)
# ══════════════════════════════════════════════════════════

# ── 설정 ──────────────────────────────────────────────────
TS       = "20260530_150704"   # mapv3.py 에서 생성된 타임스탬프
GRID_RES = 5.0                 # m/cell

# K2 전차 제원
K2_HEIGHT_M  = 2.4    # 전고 (m) — 능선 노출 판단 기준
K2_WIDTH_M   = 3.6    # 전폭 (m) — 엄폐 판단 참고

# 능선 리스크: 주변보다 얼마나 높으면 "노출"로 볼 것인가
RIDGE_RADIUS_CELLS = 4          # 주변 탐색 반경 (4셀 = 20m)
RIDGE_EXPOSE_THR   = K2_HEIGHT_M  # 2.4m 이상 높으면 능선 노출

# 개활지 리스크: 엄폐 밀도 계산 반경
OPEN_RADIUS_CELLS  = 6          # 6셀 = 30m 반경 내 엄폐물 밀도
OPEN_MIN_COVER     = 0.05       # 엄폐물 5% 미만이면 완전 개활지

# 가중치 (최종 코스트 합산용)
W_BASE   = 1.0
W_RIDGE  = 3.0   # 능선 노출 — 전차는 피탐지가 치명적
W_OPEN   = 2.0   # 개활지 — 능선보단 덜하지만 중요


# ══════════════════════════════════════════════════════════
#  리스크 계산 함수
# ══════════════════════════════════════════════════════════

def make_ridge_risk(hm: np.ndarray, radius: int = RIDGE_RADIUS_CELLS,
                    expose_thr: float = RIDGE_EXPOSE_THR) -> np.ndarray:
    """
    능선 노출 리스크
    - 현재 셀 높이 - 주변 평균 높이 > expose_thr  →  능선 위
    - 값 범위: 0.0 ~ 1.0  (1 = 완전 노출)
    
    원리:
      주변 평균보다 K2 전고(2.4m) 이상 높은 셀은
      적에게 실루엣이 보일 가능성이 높음
    """
    # 주변 평균 (uniform_filter = 박스 평균)
    kernel = 2 * radius + 1
    local_mean = uniform_filter(hm, size=kernel)

    # 현재 셀이 주변 평균보다 얼마나 높은가
    height_above = hm - local_mean

    # 0~1 정규화 (expose_thr 기준, 최대 2배까지 선형)
    ridge_raw = np.clip(height_above / expose_thr, 0.0, None)
    ridge_risk = np.clip(ridge_raw / 2.0, 0.0, 1.0)  # 2배(4.8m) 이상이면 최대

    return ridge_risk


def make_open_risk(hm: np.ndarray, obstacle: np.ndarray,
                   cost_map: np.ndarray,
                   radius: int = OPEN_RADIUS_CELLS,
                   min_cover: float = OPEN_MIN_COVER) -> np.ndarray:
    """
    개활지 노출 리스크
    - 주변 N셀 내 엄폐 가능 지형 밀도가 낮을수록 리스크↑
    - 엄폐 가능 = 장애물 셀 OR 급경사(inf cost) 셀
    - 값 범위: 0.0 ~ 1.0  (1 = 완전 개활지)
    
    원리:
      30m 반경 내 엄폐물(건물, 절벽 등)이 없으면
      사방에서 탐지/사격에 노출됨
    """
    # 엄폐 가능한 셀 = 장애물 OR 통과 불가 경사
    cover_mask = obstacle.astype(float)
    cover_mask[np.isinf(cost_map)] = 1.0

    # 주변 엄폐 밀도 (0~1)
    kernel = 2 * radius + 1
    cover_density = uniform_filter(cover_mask, size=kernel)

    # 밀도가 낮을수록 리스크 높음
    # min_cover(5%) 이하면 최대 리스크, 20% 이상이면 0
    open_risk = 1.0 - np.clip(
        (cover_density - min_cover) / (0.20 - min_cover), 0.0, 1.0
    )

    return open_risk


def make_combined_risk_cost(base_cost: np.ndarray,
                             ridge_risk: np.ndarray,
                             open_risk: np.ndarray) -> np.ndarray:
    """
    최종 통합 코스트
    - inf 셀(통과 불가)은 그대로 유지
    - 나머지: base + 리스크 가중합
    """
    combined = (W_BASE  * base_cost
              + W_RIDGE * ridge_risk
              + W_OPEN  * open_risk)

    # 통과 불가 셀 보존
    combined[np.isinf(base_cost)] = np.inf

    return combined


# ══════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════

def main():
    print(f"[로드] {TS} 맵 파일 로딩...")
    hm       = np.load(f"heightmap_{TS}_filled_final.npy")
    obstacle = np.load(f"obstacle_{TS}_final.npy")
    cost_v3  = np.load(f"cost_map_{TS}_final.npy")
    slope    = np.load(f"slope_{TS}_final.npy")

    ROWS, COLS = hm.shape
    print(f"  맵 크기: {ROWS}×{COLS}  ({ROWS*GRID_RES:.0f}m × {COLS*GRID_RES:.0f}m)")

    # ── 리스크 계산 ───────────────────────────────────────
    print("[계산] 능선 리스크...")
    ridge_risk = make_ridge_risk(hm)

    print("[계산] 개활지 리스크...")
    open_risk  = make_open_risk(hm, obstacle, cost_v3)

    print("[계산] 통합 코스트...")
    combined   = make_combined_risk_cost(cost_v3, ridge_risk, open_risk)

    # ── 통계 ─────────────────────────────────────────────
    passable = ~np.isinf(combined)
    print(f"\n[통계]")
    print(f"  능선 리스크 — 평균: {ridge_risk[passable].mean():.3f}  "
          f"최대: {ridge_risk[passable].max():.3f}")
    print(f"  개활지 리스크 — 평균: {open_risk[passable].mean():.3f}  "
          f"최대: {open_risk[passable].max():.3f}")
    print(f"  기존 코스트 범위: {cost_v3[passable].min():.1f} ~ "
          f"{np.percentile(cost_v3[passable], 95):.1f}")
    print(f"  통합 코스트 범위: {combined[passable].min():.2f} ~ "
          f"{np.percentile(combined[passable], 95):.2f}")
    print(f"  통과불가 셀: {int(np.isinf(combined).sum())} "
          f"({np.isinf(combined).mean()*100:.1f}%)")

    # ── 저장 ─────────────────────────────────────────────
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f"ridge_risk_{ts_now}.npy",    ridge_risk)
    np.save(f"open_risk_{ts_now}.npy",     open_risk)
    np.save(f"cost_risk_v1_{ts_now}.npy",  combined)
    print(f"\n[저장] ridge_risk / open_risk / cost_risk_v1 — {ts_now}")

    # ── 시각화 ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    fig.suptitle(f"K2 전차 리스크 레이어 v1  ({ts_now})", fontsize=14)

    def show(ax, data, title, cmap, vmin=None, vmax=None, label=""):
        finite = data[np.isfinite(data)]
        vm = vmax if vmax is not None else (float(finite.max()) if len(finite) else 1)
        im = ax.imshow(np.where(np.isinf(data), np.nan, data),
                       origin="lower", cmap=cmap, aspect="equal",
                       vmin=vmin, vmax=vm)
        # 장애물/inf → 검정
        if np.isinf(data).any():
            ax.imshow(np.where(np.isinf(data), 1, np.nan),
                      origin="lower", cmap="gray", aspect="equal",
                      alpha=0.9, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (cell)"); ax.set_ylabel("Z (cell)")
        plt.colorbar(im, ax=ax, label=label)

    # 상단: 입력 레이어
    show(axes[0,0], hm,      "HeightMap (m)",         "terrain",   label="고도 (m)")
    show(axes[0,1], slope,   "Slope Map (°)",          "hot_r",    vmax=45, label="경사도 (°)")
    show(axes[0,2], cost_v3, "Base Cost Map (기존)",   "RdYlGn_r", vmax=20, label="비용")

    # 하단: 리스크 레이어 + 통합
    show(axes[1,0], ridge_risk, "Ridge Risk\n(능선 노출 피탐지)", "YlOrRd",
         vmin=0, vmax=1, label="리스크 (0~1)")
    show(axes[1,1], open_risk,  "Open Risk\n(개활지 노출)",       "YlOrRd",
         vmin=0, vmax=1, label="리스크 (0~1)")
    show(axes[1,2], combined,   "Combined Cost\n(A* 입력용)",     "RdYlGn_r",
         vmax=None, label="통합 비용")

    plt.tight_layout()
    fname = f"risk_layer_v1_{ts_now}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {fname}")

    # ── 경로 비교용 차이맵 ────────────────────────────────
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    diff = np.where(np.isinf(combined), np.nan,
                    combined - np.where(np.isinf(cost_v3), 0, cost_v3))
    im = ax2.imshow(diff, origin="lower", cmap="hot", aspect="equal")
    ax2.set_title("리스크 추가분\n(Combined - Base Cost)", fontsize=12)
    ax2.set_xlabel("X (cell)"); ax2.set_ylabel("Z (cell)")
    plt.colorbar(im, ax=ax2, label="추가된 비용")
    plt.tight_layout()
    fname2 = f"risk_diff_{ts_now}.png"
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {fname2}")

    print(f"\n✅ 완료! global_planner.py 에서 다음 파일을 사용하세요:")
    print(f"   cost_map_{TS}_final.npy  →  cost_risk_v1_{ts_now}.npy")

    return ridge_risk, open_risk, combined, ts_now


if __name__ == "__main__":
    main()