"""
맵 입출력 — 로드 / 저장 / 타임스탬프 / 최신본 자동탐지

이전에는 np.load 4줄이 risklayer, risk_v2, planner에 반복됐음.
MapBundle 하나로 묶어서 한 번에 로드.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from . import config


@dataclass
class MapBundle:
    """한 타임스탬프의 모든 맵 레이어 묶음"""
    ts: str
    heightmap: np.ndarray          # 실측 보정본 (_final)
    heightmap_filled: np.ndarray   # 보간 채움본 (_filled_final)
    slope: np.ndarray              # 경사도 (°)
    cost: np.ndarray               # base cost (v3)
    obstacle: np.ndarray           # 장애물 bool

    @property
    def shape(self):
        return self.cost.shape


def timestamp() -> str:
    """현재 시각 타임스탬프"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_ts() -> str:
    """
    DATA_DIR에서 가장 최근 cost_map_*_final.npy 의 타임스탬프 반환.
    없으면 FileNotFoundError.
    """
    pat = re.compile(r"cost_map_(\d{8}_\d{6})_final\.npy$")
    found = []
    for p in config.DATA_DIR.glob("cost_map_*_final.npy"):
        m = pat.search(p.name)
        if m:
            found.append(m.group(1))
    if not found:
        raise FileNotFoundError(
            f"{config.DATA_DIR} 에 cost_map_*_final.npy 가 없습니다. "
            "먼저 mapping 단계를 실행하세요."
        )
    return sorted(found)[-1]


def _load(name: str) -> np.ndarray:
    return np.load(config.DATA_DIR / name)


def load_maps(ts: str | None = None) -> MapBundle:
    """
    *_final.npy 세트를 MapBundle로 로드.
    ts=None 이면 최신본 자동 선택.
    """
    if ts is None:
        ts = latest_ts()
    return MapBundle(
        ts               = ts,
        heightmap        = _load(f"heightmap_{ts}_final.npy"),
        heightmap_filled = _load(f"heightmap_{ts}_filled_final.npy"),
        slope            = _load(f"slope_{ts}_final.npy"),
        cost             = _load(f"cost_map_{ts}_final.npy"),
        obstacle         = _load(f"obstacle_{ts}_final.npy"),
    )


def save_array(name: str, arr: np.ndarray) -> str:
    """DATA_DIR에 .npy 저장. 전체 경로 문자열 반환."""
    path = config.DATA_DIR / name
    np.save(path, arr)
    return str(path)


def resolve_cost(ts: str | None = None, prefer: str = "risk"):
    """
    planner 입력 cost 맵 선택.
      prefer="risk" : cost_risk_v2_*.npy(최신) 우선, 없으면 base cost로 폴백
      prefer="base" : 항상 base cost
    반환: (cost_array, source_label)
    """
    bundle = load_maps(ts)

    if prefer == "risk":
        # 기존 코드는 v2가 하나라도 있으면 항상 v2만 우선해서,
        # 방금 만든 v1 risk cost가 있어도 오래된 v2를 잡는 문제가 있었음.
        # 이제 v1/v2를 모두 모은 뒤 파일명 끝의 타임스탬프 기준 최신 것을 사용한다.
        risk_files = (list(config.DATA_DIR.glob("cost_risk_v2_*.npy"))
                      + list(config.DATA_DIR.glob("cost_risk_v1_*.npy")))

        def _risk_ts(path):
            m = re.search(r"cost_risk_v[12]_(\d{8}_\d{6})\.npy$", path.name)
            return m.group(1) if m else ""

        risk_files = sorted(risk_files, key=_risk_ts)
        if risk_files:
            cost = np.load(risk_files[-1])
            return cost, f"risk:{risk_files[-1].name}"

    return bundle.cost, f"base:cost_map_{bundle.ts}_final.npy"
