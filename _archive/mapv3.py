import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, gaussian_filter
from datetime import datetime

TS       = "20260530_142831"
GRID_RES = 5

def correct(arr):
    """최종 보정: 단순 전치(Transpose)"""
    return arr.T   # flipud(arr.T)에서 flipud 한번 더 = arr.T

# ── 로드 ──────────────────────────────────────────
hm_raw  = np.load(f"heightmap_{TS}.npy")
hm_fill = np.load(f"heightmap_{TS}_filled.npy")
obs     = np.load(f"obstacle_{TS}.npy")

# ── 보정 ──────────────────────────────────────────
hm_raw_c  = correct(hm_raw)
hm_fill_c = correct(hm_fill)
obs_c     = correct(obs)

# ── cost v3 재생성 ─────────────────────────────────
def make_cost_v3(hm, obs):
    hm_smooth = gaussian_filter(hm, sigma=1.0)
    gx, gz    = np.gradient(hm_smooth, GRID_RES)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gz**2)))

    cost = np.ones_like(slope_deg)
    cost[slope_deg > 10] = 2.0
    cost[slope_deg > 20] = 5.0
    cost[slope_deg > 30] = np.inf

    from scipy.ndimage import maximum_filter, minimum_filter
    diff = maximum_filter(hm, size=3) - minimum_filter(hm, size=3)
    cost[diff > 5.0]           = np.inf
    cost[obs.astype(bool)]     = np.inf

    inflated = binary_dilation(np.isinf(cost), iterations=2)
    cost[inflated & ~np.isinf(cost)] = 8.0

    return slope_deg, cost

slope_v3, cost_v3 = make_cost_v3(hm_fill_c, obs_c)

# ── 저장 ──────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
np.save(f"heightmap_{ts}_final.npy",      hm_raw_c)
np.save(f"heightmap_{ts}_filled_final.npy", hm_fill_c)
np.save(f"slope_{ts}_final.npy",          slope_v3)
np.save(f"cost_map_{ts}_final.npy",       cost_v3)
np.save(f"obstacle_{ts}_final.npy",       obs_c)

# ── 시각화 ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

im0 = axes[0].imshow(hm_fill_c, origin="lower", cmap="terrain", aspect="equal")
axes[0].set_title("HeightMap (최종)")
axes[0].set_xlabel("X축"); axes[0].set_ylabel("Z축")
plt.colorbar(im0, ax=axes[0], label="고도 (m)")

im1 = axes[1].imshow(slope_v3, origin="lower", cmap="hot_r",
                     aspect="equal", vmin=0, vmax=45)
axes[1].set_title("Slope Map v3 (최종)")
axes[1].set_xlabel("X축"); axes[1].set_ylabel("Z축")
plt.colorbar(im1, ax=axes[1], label="경사도 (°)")

cost_vis = np.where(np.isinf(cost_v3), 20, cost_v3)
im2 = axes[2].imshow(cost_vis, origin="lower", cmap="RdYlGn_r",
                     aspect="equal", vmin=1, vmax=20)
axes[2].set_title("Cost Map v3 (최종 + Inflation)")
axes[2].set_xlabel("X축"); axes[2].set_ylabel("Z축")
plt.colorbar(im2, ax=axes[2], label="비용")

obs_vis = np.ma.masked_where(~obs_c.astype(bool), obs_c.astype(float))
for ax in axes:
    ax.imshow(obs_vis, origin="lower", cmap="cool", aspect="equal", alpha=0.8)

plt.suptitle(f"All Layers Final — {ts}", fontsize=14)
plt.tight_layout()
plt.savefig(f"all_layers_final_{ts}.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ 최종 저장 완료: {ts}")
for name in ["heightmap","heightmap_filled","slope","cost_map","obstacle"]:
    print(f"   {name}_{ts}_final.npy")