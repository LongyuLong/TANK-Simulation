import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ── 로드 ────────────────────────────────────────────
hm_filled    = np.load("heightmap_20260530_142831_filled.npy")
obstacle_map = np.load("obstacle_20260530_142831.npy")

# ── 개선된 경사도 + 비용맵 ───────────────────────────
def make_cost_v2(hm, obs, grid_res=5):
    grad_x, grad_z = np.gradient(hm, grid_res)
    slope_deg      = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_z**2)))

    cost = np.ones_like(slope_deg)
    cost[slope_deg > 10]  = 2.0
    cost[slope_deg > 20]  = 5.0
    cost[slope_deg > 30]  = np.inf   # 탱크 통과 불가

    # 장애물 inf
    cost[obs] = np.inf

    # 절벽 감지: 주변 셀과 높이 차이가 큰 곳
    from scipy.ndimage import maximum_filter, minimum_filter
    local_max = maximum_filter(hm, size=3)
    local_min = minimum_filter(hm, size=3)
    height_diff = local_max - local_min
    cliff_mask  = height_diff > 5.0   # 5m 이상 차이 = 절벽
    cost[cliff_mask] = np.inf

    return slope_deg, cost

slope_v2, cost_v2 = make_cost_v2(hm_filled, obstacle_map)

# ── 저장 ────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
np.save(f"slope_v2_{ts}.npy",    slope_v2)
np.save(f"cost_map_v2_{ts}.npy", cost_v2)

# ── 시각화 ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

im0 = axes[0].imshow(hm_filled, origin="lower", cmap="terrain", aspect="equal")
axes[0].set_title("HeightMap (filled)")
plt.colorbar(im0, ax=axes[0], label="고도 (m)")

im1 = axes[1].imshow(slope_v2, origin="lower", cmap="hot_r",
                      aspect="equal", vmin=0, vmax=45)
axes[1].set_title("Slope Map v2")
plt.colorbar(im1, ax=axes[1], label="경사도 (°)")

cost_vis = np.where(np.isinf(cost_v2), 20, cost_v2)
im2 = axes[2].imshow(cost_vis, origin="lower", cmap="RdYlGn_r", aspect="equal",
                      vmin=1, vmax=20)
axes[2].set_title("Cost Map v2 (A* 입력)\n빨강=통과불가 / 노랑=고비용 / 초록=저비용")
plt.colorbar(im2, ax=axes[2], label="비용")

plt.suptitle(f"Cost Map v2 — 절벽+경사도 개선", fontsize=14)
plt.tight_layout()
plt.savefig(f"cost_map_v2_{ts}.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 통계 출력 ────────────────────────────────────────
total    = cost_v2.size
inf_cnt  = int(np.isinf(cost_v2).sum())
hi_cnt   = int(((cost_v2 >= 3) & ~np.isinf(cost_v2)).sum())
free_cnt = int((cost_v2 < 3).sum())

print(f"💾 cost_map_v2_{ts}.npy 저장")
print(f"   통과불가 (inf) : {inf_cnt:4d} 셀 ({inf_cnt/total*100:.1f}%)")
print(f"   고비용 (≥3)   : {hi_cnt:4d} 셀 ({hi_cnt/total*100:.1f}%)")
print(f"   자유 (<3)     : {free_cnt:4d} 셀 ({free_cnt/total*100:.1f}%)")
print(f"\n다음 단계: cost_map_v2_{ts}.npy 를 A* 입력으로 사용")