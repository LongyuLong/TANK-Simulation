"""
tanknav — K2 전차 자율주행 내비게이션 패키지

파이프라인:
  Yaxis.py (수집, 패키지 외부 Flask 서버)
      ↓ heightmap / obstacle (raw)
  mapping   : 보정 + base cost (v3)
      ↓ *_final.npy
  risk      : ridge / open / LoS 리스크 통합
      ↓ cost_risk_*.npy
  planning  : A* 전역 경로

확장 예정:
  perception   : 비전 지형분류 (물/숲/바위) → mapping·risk 주입
  local        : 로컬 플래너 (DWA 등) ← planning 경로 입력
  fire_control : 사격통제 ← risk 적정보 공유
"""
from . import config, mapio, viz, mapping, risk, planning, perception

__all__ = ["config", "mapio", "viz", "mapping", "risk", "planning", "perception"]
