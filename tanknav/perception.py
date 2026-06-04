"""
[비전 - 확장 예정]  탑뷰 지형 분류 (물 / 숲 / 바위)

게임 탑뷰 스크린샷에서 색상(HSV) 기반으로 통행 위험 지형을 분류하여
mapping.make_cost(extra_inf=...) 또는 risk 레이어에 주입하기 위한 모듈.

현재는 인터페이스 스텁만. 구현 시:
  classify(image) -> dict[str, np.ndarray]  (각 클래스별 bool 마스크)
  to_extra_inf(masks) -> np.ndarray         (물 등 통과불가 합산)
  to_risk(masks)      -> np.ndarray         (숲 등 0~1 리스크)
"""
from __future__ import annotations
import numpy as np


def classify(image: np.ndarray) -> dict[str, np.ndarray]:
    """탑뷰 이미지 → 클래스별 마스크. (미구현)"""
    raise NotImplementedError(
        "perception.classify: 비전 지형분류 미구현. "
        "탑뷰 스크린샷이 준비되면 HSV 색상범위 분류로 구현 예정."
    )


def to_extra_inf(masks: dict[str, np.ndarray]) -> np.ndarray:
    """통과불가 지형(깊은 물 등) 합산 bool 마스크. (미구현)"""
    raise NotImplementedError
