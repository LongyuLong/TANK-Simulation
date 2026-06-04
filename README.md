# TANK Simulation — 위협-인지 전차 글로벌 패스 플래너

K2 전차의 **전역 경로**를 지형·적 위협을 고려해 생성하는 플래너.
교리 근거: 미 육군 **ATP 3-20.15 Tank Platoon**, OAKOC.

```
지형/적 위협 → 채점기(탐지→피격 체인, 생존확률) → threat_cost로 A* → Theta* 스무싱 → 경로
```

## 핵심 개념

- **채점기(scorer)**: 경로의 "좋음"을 `P(생존)`으로 정량화. `P(피격)=P(탐지)·P(교전|탐지)·P(명중|교전)·P(살상|명중)`, 탐지는 `is_visible`(지형차폐 LoS)로 직접 판정.
- **Option S (채점기 직결)**: 플래너 비용 = `threat_cost = base + W_SURV·hazard`, `hazard=−log(1−p_hit)`. A* 비용 최소화 = **생존 최대화**.
- **Theta\* 스무싱**: any-angle 직선화 — 단, **비용 적분 게이트**로 위험구역 직선관통 차단.
- **가상 돌(object) 레이어**: 통과불가(inf)+마진, 개활지 리스크 감소, LoS 차폐(los_surface).

## 격자
61×61, `GRID_RES=5m`, 맵 300×300m. 좌표 (row=Z, col=X).

## 구조
```
tanknav/
  config.py      파라미터 단일 소스 (사거리·속도·가중치)
  mapio.py       맵 로드(MapBundle)
  mapping.py     base cost + inflation + 가상 돌(apply_virtual_rocks)
  risk.py        적 모델 Enemy, is_visible(LoS), ridge/open/los_risk
  eval.py        채점기: precompute_threat / score_path / threat_cost / hazard_field
  planning.py    A* + Theta*(비용-인지), plan_threat(Option S)
  viz.py         시각화 헬퍼
run.py           CLI (cost / risk / plan / all / threat)
score_run.py · score_sensitivity.py · scenarios.py   실험 스크립트
```

## 빠른 시작
```bash
# 환경: numpy / scipy / matplotlib
python run.py threat --start 5 5 --goal 55 55       # 위협-인지 계획(적 터미널 입력)
python run.py threat --start 5 5 --goal 55 55 --w-surv 40   # 생존 더 우선
python run.py plan   --start 5 5 --goal 55 55       # 휴리스틱 cost_risk (fallback)
```
코드:
```python
from tanknav import planning, risk
enemies = [risk.Enemy("tank", [(30,30)]),
           risk.Enemy("patrol", risk.bresenham(45,10,45,50))]
planning.plan_threat((5,5), (55,55), enemies)   # config.VIRTUAL_ROCKS 자동 반영
```
출력: `global_path_<ts>.png`(heightmap | threat intensity | threat_cost), `path_<ts>.npy`.

## 주요 노브 (config.py)
| 이름 | 의미 |
|---|---|
| `W_SURV_PLAN` | 생존↔시간 (클수록 우회↑) |
| `THETA_COST_TOL` | 스무싱 코너컷 허용 (0=위험증가 금지) |
| `DETECT_RANGE`/`WEAPON_RANGE` | 적 탐지/무기 사거리 (sim 제원) |
| `VIRTUAL_ROCKS` | 가상 돌 배치 |

## 상태 / 로드맵
- ✅ 채점기 + Option S(채점기 직결) + 비용-인지 Theta* + 가상 돌(LoS 차폐)
- ⏳ cover(defilade, 차체차폐) / T_EXPOSE 정식화 / D*(증분 재계획) / 스케일 라벨 확정

자세한 설계·인수인계: [`HANDOFF_YUSUNG.md`](HANDOFF_YUSUNG.md), [`MERGE_S_OPTION.md`](MERGE_S_OPTION.md)
