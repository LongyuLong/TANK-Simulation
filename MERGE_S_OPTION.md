# 머지 노트 — Option S (채점기 직결 글로벌 계획) 통합

> 대상: 글로벌 패스 팀. 이 브랜치(rocks + Theta*)에 **채점기 직결 비용(Option S)** 을 통합함.
> 배경·수식 상세: `HANDOFF_YUSUNG.md`.

## 결정 (A1)
글로벌 플래너 기본 비용 = **`threat_cost`(채점기 직결)** 로 확정.
- 근거: `precompute_threat` 측정 **280ms/미션**(적 정적 → 미션당 1회) → 휴리스틱 대리지표의 속도 이점 소멸.
- `threat_cost = base + W_SURV_PLAN·hazard`, `hazard=−log(1−p_hit)`, `Σhazard=−log(P_survive)`
  → A* 비용합 최소화 = **생존 최대화**. 더 원리적·정직(is_visible 기반).
- 휴리스틱 `cost_risk`(ΣW·risk)는 **fallback으로 보존**(삭제 안 함). `tune.py`는 선반행.

## 변경 파일
| 파일 | 변경 |
|---|---|
| `tanknav/config.py` | `W_SURV_PLAN=20.0` 추가, `THETA_COST_TOL=0.0` 추가, urgency 설계 주석(미구현) |
| `tanknav/eval.py` | `precompute_threat(..., hm_los)` — 돌 LoS 차폐면(los_surface) 반영 / `hazard_field`, `threat_cost` 추가 |
| `tanknav/planning.py` | `theta_smooth_path` **비용-인지 게이트**(P1), `path_stats` **실제 통과셀**(P6), `plan_threat()` + `_plot_threat()` 추가 |
| `run.py` | `threat` 서브커맨드 추가 |

## 버그 수정
- **P1**: Theta* 스무싱이 inf만 보고 위험구역 직선관통 → 비용 적분 게이트로 차단.
  실측: smooth 위험 186→**300(+61%)** → 수정 후 **187(+0.8%)**.
- **P6**: `path_stats`가 waypoint 셀만 합산 → 통과셀 supercover 합산으로 정정(착시 제거).

## 사용법
```bash
# 위협-인지 계획 (적은 터미널 입력: tank x z / patrol x1 z1 x2 z2 ... / done)
python run.py threat --start 5 5 --goal 55 55
python run.py threat --start 5 5 --goal 55 55 --w-surv 40   # 생존 더 우선(우회↑)

# 코드에서
from tanknav import planning, risk
enemies = [risk.Enemy("tank",[(30,30)]), risk.Enemy("patrol", risk.bresenham(45,10,45,50))]
planning.plan_threat((5,5),(55,55), enemies)   # 돌(config.VIRTUAL_ROCKS) 자동 반영
```
출력: `global_path_<ts>.png`(heightmap | threat intensity | threat_cost), `path_<ts>.npy`(스무싱 경로, 미터).

## 노브
- `W_SURV_PLAN`(config): 생존↔시간. 클수록 우회↑.
- `THETA_COST_TOL`(config): 스무싱 코너컷 허용. 0.0=위험증가 금지.
- (설계) `urgency` 하나로 두 노브 묶기 — config 주석 참조, A1 확정됐으니 배선 가능.

## 후속(합의된 보류)
- **P4 cover(defilade)**: 지금 돌은 LoS 차폐(los_surface)만 반영. cover는 open_risk 프록시 → 차체높이 가림 판정으로 교체 예정.
- **T_EXPOSE**: 절대 생존 스케일 노브, hazard 정식화 후속.
- **D\***: 로컬 미지장애물 인식 단계 후 증분 재계획 도입(계획 유효, 현재 정적 A*+Theta*).
- **스케일 라벨**: 130≙~500m, LiDAR(=적 열상) 80≙~300m 로 문서 확정 예정.
