"""
공통 시각화 헬퍼

이전에 risklayer / risk_v2 에 중복됐던 show() 를 한 곳으로.
matplotlib는 Agg 백엔드(파일 저장 전용)로 강제.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np

from . import config

# 한글 폰트: koreanize-matplotlib 있으면 사용, 없으면 설치된 한글 폰트 탐색.
def _setup_korean_font():
    try:
        import koreanize_matplotlib  # noqa: F401  (import만으로 자동 설정)
        return
    except ImportError:
        pass
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_korean_font()


def show(ax, data, title, cmap, vmin=None, vmax=None, label=""):
    """
    한 축에 맵 1장 그리기.
    - inf(통과 불가) 셀은 회색으로 오버레이
    - vmax 미지정 시 유한값 최대로 자동
    """
    finite = data[np.isfinite(data)]
    vm = vmax if vmax is not None else (float(finite.max()) if finite.size else 1)
    im = ax.imshow(np.where(np.isinf(data), np.nan, data),
                   origin="lower", cmap=cmap, aspect="equal",
                   vmin=vmin, vmax=vm)
    if np.isinf(data).any():
        ax.imshow(np.where(np.isinf(data), 1, np.nan),
                  origin="lower", cmap="gray", aspect="equal",
                  alpha=0.9, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (cell)")
    ax.set_ylabel("Z (cell)")
    plt.colorbar(im, ax=ax, label=label)
    return im


def draw_path(ax, path):
    """경로(list of (row,col))를 축에 오버레이"""
    if not path:
        return
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, "b-", linewidth=2, label="path")
    ax.plot(xs[0],  ys[0],  "go", markersize=10, label="start")
    ax.plot(xs[-1], ys[-1], "r*", markersize=12, label="goal")
    ax.legend(fontsize=8, loc="upper right")


def savefig(fig, name: str, dpi: int = 150) -> str:
    """DATA_DIR에 그림 저장 후 닫기. 전체 경로 반환."""
    path = config.DATA_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)




def _rock_center_to_cell_for_plot(rock: dict) -> tuple[float, float]:
    """config의 돌 좌표를 plot용 (row, col) 셀 좌표로 변환한다."""
    if "row" in rock and "col" in rock:
        return float(rock["row"]), float(rock["col"])
    if "z" in rock and "x" in rock:
        return float(rock["z"]) / config.GRID_RES, float(rock["x"]) / config.GRID_RES
    raise ValueError(f"rock에는 row/col 또는 x/z가 필요합니다: {rock}")


def _stable_rock_seed(rock: dict) -> int:
    """실행할 때마다 같은 모양이 나오도록 돌 설정에서 결정적 seed를 만든다."""
    name = str(rock.get("name", "rock"))
    row, col = _rock_center_to_cell_for_plot(rock)
    text = f"{name}:{row:.3f}:{col:.3f}:{rock.get('radius_m', 0)}"
    seed = 0
    for ch in text:
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    return seed


def draw_rock_symbols(ax, rocks=None, label="virtual rocks", show_label=True):
    """
    가상 큰돌을 하늘색 사각형 대신 지도용 '돌 심볼'로 표시한다.

    - 돌 본체는 회갈색 불규칙 다각형
    - 좌하단에는 그림자
    - 상단에는 밝은 면을 추가해 바위처럼 보이게 함
    - 실제 경로계획 mask는 그대로 유지하고, 시각화만 개선한다.
    """
    rocks = rocks if rocks is not None else getattr(config, "VIRTUAL_ROCKS", [])
    if not rocks:
        return

    first = True
    for rock in rocks:
        row, col = _rock_center_to_cell_for_plot(rock)
        radius_cells = max(0.8, float(rock.get("radius_m", config.GRID_RES)) / config.GRID_RES)

        rng = np.random.default_rng(_stable_rock_seed(rock))
        n = 11
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        jitter = rng.uniform(0.78, 1.16, size=n)

        # 살짝 납작한 돌 형태
        x_scale = radius_cells * 1.05
        y_scale = radius_cells * 0.82
        xs = col + np.cos(angles) * x_scale * jitter
        ys = row + np.sin(angles) * y_scale * jitter
        points = np.column_stack([xs, ys])

        # 그림자
        shadow = points + np.array([0.22 * radius_cells, -0.20 * radius_cells])
        ax.add_patch(
            Polygon(
                shadow,
                closed=True,
                facecolor="#2b241d",
                edgecolor="none",
                alpha=0.35,
                zorder=6,
            )
        )

        # 돌 본체
        body_label = label if (first and show_label) else None
        ax.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor="#77736b",
                edgecolor="#302e2a",
                linewidth=1.1,
                alpha=0.96,
                zorder=7,
                label=body_label,
            )
        )

        # 밝은 상단 면
        top_points = np.column_stack([
            col + np.cos(angles) * x_scale * jitter * 0.63 - 0.12 * radius_cells,
            row + np.sin(angles) * y_scale * jitter * 0.48 + 0.18 * radius_cells,
        ])
        ax.add_patch(
            Polygon(
                top_points,
                closed=True,
                facecolor="#aaa69b",
                edgecolor="#5c5952",
                linewidth=0.7,
                alpha=0.85,
                zorder=8,
            )
        )

        # 균열선 느낌
        crack_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        crack_len = radius_cells * 0.7
        ax.plot(
            [col - 0.12 * radius_cells, col + np.cos(crack_angle) * crack_len],
            [row + 0.12 * radius_cells, row + np.sin(crack_angle) * crack_len * 0.55],
            color="#4a4741",
            linewidth=0.7,
            alpha=0.75,
            zorder=9,
        )
        first = False


def draw_rock_cover_zones(ax, rocks=None, cover_cells=None, label="rock cover"):
    """
    돌 주변 엄폐 영향권을 네모 윤곽선 대신 은은한 원형 점선으로 표시한다.
    """
    rocks = rocks if rocks is not None else getattr(config, "VIRTUAL_ROCKS", [])
    cover_cells = int(
        cover_cells if cover_cells is not None
        else getattr(config, "ROCK_COVER_CELLS", 0)
    )
    if not rocks or cover_cells <= 0:
        return

    first = True
    for rock in rocks:
        row, col = _rock_center_to_cell_for_plot(rock)
        radius_cells = max(0.8, float(rock.get("radius_m", config.GRID_RES)) / config.GRID_RES)
        total_radius = radius_cells + cover_cells
        ax.add_patch(
            Circle(
                (col, row),
                radius=total_radius,
                fill=False,
                edgecolor="#6f8060",
                linewidth=0.9,
                linestyle=(0, (3, 3)),
                alpha=0.7,
                zorder=5,
                label=label if first else None,
            )
        )
        first = False

def draw_mask_outline(ax, mask, color="cyan", label="virtual rocks", linewidth=1.8, linestyle="-"):
    """bool mask를 지도 위에 윤곽선으로 표시한다. 큰돌 위치 확인용."""
    if mask is None:
        return
    arr = np.asarray(mask, dtype=float)
    if not np.any(arr):
        return
    # contour는 True/False 경계선을 그려서 cost map 위에 돌 위치를 보이게 한다.
    ax.contour(arr, levels=[0.5], colors=[color], linewidths=linewidth, linestyles=linestyle, origin="lower")
    # 범례용 더미 라인
    ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)
