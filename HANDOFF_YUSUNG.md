# HANDOFF — 전차 Global Path 채점기 (tanknav/eval.py)

> 이 문서는 **다른 LLM에 그대로 입력해 작업을 이어받기 위한** 인수인계서다.
> 프로젝트: K2 전차 전역 경로계획. 목표 = A* 경로의 "좋음"을 객관 채점하는
> 목적함수를 만들고, 그걸로 비용맵 가중치를 자동 튜닝(Option A).
> 교리 근거: 미 육군 **ATP 3-20.15 Tank Platoon (2025)**, OAKOC.

---

## 0. 이어받는 LLM에게 (먼저 읽을 것)

- **환경**: Windows, 코드 실행은 conda env `tank`.
  실행 예: `& C:\Users\acorn\anaconda3\envs\tank\python.exe score_run.py <label>`
- **격자**: 61×61, `GRID_RES=5m`, 맵 300m×300m (대각 ~424m). 좌표는 (row=Z, col=X) 셀.
- **가장 중요한 원칙 (순환논리 금지)**: 채점기는 비용맵 휴리스틱(`ridge/open/los_risk`
  가중합)과 **레벨이 달라야 한다**. 비용맵 = "위험할 것 같다"는 대리지표. 채점기 =
  "실제 탐지·피격됐는가"라는 ground truth. 그래서 채점기는 `is_visible()`로 LoS를
  직접 재계산하고 저장된 `los_risk.npy`를 끌어다 쓰지 않는다.
- **현재 단계**: 채점기 v1 구현·검증 완료. 시간 항은 비활성(생존 모델에 집중 중).
  다음은 "채점기로 재계획" → "object 심고 cover 부활".

---

## 1. 채점 모델 (확정 수식)

경로를 세그먼트(셀→다음셀) 단위로 보고:

```
P(피격_seg) = intensity · concealment · exposure · PK_BASE · cover
P(생존)     = exp( Σ_seg log(1 − P(피격_seg)) )        # 로그합 = 수치안정
score       = w_surv·(1 − P생존) + w_time·t_norm        # 비용, 낮을수록 좋음
t_norm      = total_time / (직선거리 / 최대속도)
```

각 항:
- **intensity[cell]** = `max_적위치 [ p_range·p_engage·p_hit ]`, 단 `is_visible`이고
  사거리 이내일 때만. 적-의존, 경로 무관 → **precompute로 1회 계산**.
  - `p_range(d)   = max(0, 1 − d/DETECT_RANGE)`            (탐지 거리감쇠, 선형)
  - `p_engage(d)  = 2/3 규칙` (2/3선 이내 1.0, MEL까지 선형감쇠, 초과 0).
    **단 현재 sim은 DETECT(80)<2/3선(87)이라 항상 1.0** (센서 병목).
  - `p_hit(d)     = Ph_max · exp(−k·d)`                    (무기별 WEAPON_PH)
- **concealment[cell]** = `clip(1 − 0.4·ridge_risk − 0.4·open_risk, 0.1, 1.0)` (지형, fresh)
- **cover[cell]** = `COVER_FLOOR + (1−COVER_FLOOR)·open_risk` ← **v1 임시** (open_risk 할인)
- **exposure** = `1 − exp(−dt/T_EXPOSE)`, `dt = 세그먼트길이 / TIER_SPEED[tier]`
- **통행성 tier** (속도): slope < SLOPE_TIER[0] → unrestricted, < [1] → restricted,
  그 이상 → severely_restricted; obstacle이면 blocked(속도 0 = 경로 무효).

> ablation 분석 결과 **경로 순위의 1차 동력은 intensity(LoS·사각 활용)**,
> concealment는 보정(−0.13~−0.20), cover는 현재 거의 무의미(−0.00~−0.05, 개활지형),
> exposure/T_EXPOSE는 절대 생존 스케일을 지배하는 전역 노브.

---

## 2. config 파라미터 (tanknav/config.py, 현재값)

```python
# 적 제원 (sim 실제값 — LiDAR 탐지 80m가 병목, 무기 평지 130m)
DETECT_RANGE = {"infantry": 80,  "tank": 80,  "patrol": 80}   # LiDAR 확정탐지 앵커
OBS_HEIGHT   = {"infantry": 1.8, "tank": 2.4, "patrol": 2.4}
WEAPON_RANGE = {"infantry": 130, "tank": 130, "patrol": 130}  # MEL(평지). 고지↑는 v2
WEAPON_PH    = {"infantry": (0.70, 0.0020),
                "tank": (0.95, 0.0008), "patrol": (0.95, 0.0008)}  # (Ph_max, decay/m)

# 통행성 → 속도
SLOPE_TIER = (10.0, 20.0)
TIER_SPEED = {"unrestricted":15.0, "restricted":6.0,
              "severely_restricted":2.0, "blocked":0.0}

T_EXPOSE    = 5.0    # 노출 포화상수(s)
PK_BASE     = 0.7    # 명중 시 기본 살상확률
COVER_FLOOR = 0.6    # 완전엄폐 시 cover 하한 (Pk 40%↓)

# 임무 (생존가중, 시간가중) — survival = 시간 비활성
MISSION_WEIGHTS = {"survival":(1.0,0.0), "attack":(0.4,0.6),
                   "defend":(0.8,0.2), "recon":(0.9,0.1)}
```

---

## 3. 코드 인터페이스

```python
from tanknav import mapio, risk, eval as ev

bundle  = mapio.load_maps()                       # 최신 _final 맵 세트
enemies = [risk.Enemy("tank", [(40,18)]),          # etype + positions[]
           risk.Enemy("patrol", risk.bresenham(48,8,48,52))]  # patrol=정적 셀집합
precomp = ev.precompute_threat(enemies, bundle)    # Threat(intensity, concealment, cover)
result  = ev.score_path(cells, enemies, bundle, mission="survival",
                        precomp=precomp, ablate=frozenset())  # ablate: 항 중립화
# result = {P_survive, time_s, t_norm, score, n_blocked, per_seg}
```

- `risk.is_visible(hm, src_r,src_c,src_h, tgt_r,tgt_c, tgt_h)` — 적=src(관측높이),
  전차=tgt(K2 2.4m). 지형차폐 Bresenham LoS. **이미 구현됨, 재사용.**
- `ev.load_path_cells(npy)` — 저장된 `path_*.npy`(미터) → 셀 리스트.
- `ablate`: `{"concealment","cover","exposure"}` 중 일부를 1.0으로 중립화(민감도용).

---

## 4. 실행 & 결과 격리

```
python score_run.py <label>          # 채점 1회 → fin/runs/<ts>_<label>/
python score_sensitivity.py <label>  # 민감도 ablation
```
매 run = 폴더 하나에 완결 저장 (재현성). 두 스크립트 모두 `runlog.full_run()`을
호출해 **동일한 핵심 기록**을 남김:
```
fin/runs/<ts>_<label>/
  params.json      ← config 스냅샷 ("이 결과 뭐였지?" 해결)
  enemies.json     ← 적 배치
  map_info.json    ← 맵 ts/shape/위협 통계(coverage·intensity)
  paths.json       ← 채점 경로 셀+미터 좌표 (원본 npy 없이 재현)
  scores.csv/txt   ← 경로×임무 점수
  per_segment.csv  ← 셀별 위험도 분해(tier·dt·intensity·conceal·cover·exposure·p_hit)
  threat_*.npy     ← 위협층(intensity/concealment/cover) 배열
  threatmap.png    ← 위협장 + 경로 시각화
  sensitivity.*    ← (sens run만) ablation 결과
```
적 배치 변경은 `score_run.py`의 `build_enemies()` 수정.

---

## 5. 확정된 설계 결정 (바꾸지 말 것, 근거 포함)

| 결정 | 근거 |
|---|---|
| patrol = **정적 관측 집합**(시간 비동기) | 적 순찰로를 사전정보로 안다는 가정. sim 구현 무관, 튜닝 재현성 |
| 사거리 = **LiDAR 80 / 무기 130** (sim 제원) | 300m 맵에서 실제 3km는 전역 point-blank → 거리규칙 사망 |
| 2/3 규칙 **비활성** | 탐지(80)<2/3선(87): 탐지된 표적은 항상 교전권 |
| 시간 항 **비활성**(survival) | 생존 모델 집중. **계획 단계 가면 경로길이 정규화 재도입 필수**(안 그러면 사각 무한배회가 최적) |
| 80m = LiDAR **확정탐지** | 카메라(비전) 결론 나오면 불확실 탐지밴드로 확장(함수 교체) |

---

## 6. object(돌) 아키텍처 — 합의된 방향 (미구현)

**문제의식**: object layer를 맵에 추가하면 "정보가 과한가?"
**판정 기준**: ①중복성(같은 사실 2중 표현?) ②현실성(계획시점에 알 수 있나?)

**결론**:
- **정적 지형물(큰 돌) = 맵 / 동적(차·사람) = 런타임 인식** → 올바름.
- 돌 하나가 **3효과**(기동/은폐/엄폐)에 작용. 별도 semantic 레이어는 **중복**.
  → **높이를 single source of truth로**:
  ```
  ground_height (현 깨끗한 heightmap)            → slope·기동 (지면만)
  los_surface = ground_height + object_height    → is_visible·cover 전용
  돌 footprint → make_cost(extra_inf=마스크)      → cost inf + inflation
  ```
- 지면을 깨끗하게(충돌 없이) 땄으므로 돌 높이는 heightmap에 **없음** →
  절벽 자동검출(고도차>5m→inf)이 돌을 못 잡음 → **통과불가는 명시적 extra_inf로** 넣어야.

**통과불가 + 마진**: 기존 인프라가 이미 함 (`mapping.make_cost`):
`cost[obs]=inf` + `binary_dilation`(INFLATE_ITERS=2, 10m) 소프트링(INFLATE_COST=8.0).
- 마진 **soft 유지**(권고): hard로 막으면 돌 뒤 hull-down 자리에 못 붙어 cover 무용.
  soft면 기동비용↔생존이 플래너에서 trade-off로 결합.
- 마진 크기 **1셀(5m)로 축소** 검토(돌 끼고 사격 살리려면 작게).

**cover 모델 교체** (object 심은 후):
- v1 `open_risk 할인` → **defilade-delta**(partial-height LoS)로 교체.
- hull-down = "**포탑 보임(탐지O) + 차체 가려짐(Pk↓)**" = concealment와 cover가
  분리되는 유일 지점(교범 ¶2-67, ¶4-146). los_surface에서 차체높이(~1.4m) 가림 판정.

---

## 7. 보류된 개선 (선택지)

- **exposure hazard 리팩터**: 현재 식은 시간적분확률·단발명중확률 혼합 휴리스틱.
  `p_hit_seg = 1 − exp(−λ·dt)`, `λ = intensity·conceal·cover/T_EXPOSE` 로 가면
  시간적분 일관 + T_EXPOSE가 진짜 위험률 상수.
- **T_EXPOSE 스윕**: 절대 생존 스케일 민감도 파악.
- **항 정규화**: `(1−P)`(폭~0.3)와 `t_norm`(폭~1.3) 동적범위 차 → mission 가중치가
  핸들 역할 못 함. [0,1]로 맞춰야 함 (시간 항 복원 시).

---

## 8. 다음 단계 (순서)

1. ✅ 생존 모델 민감도 (완료)
2. **채점기로 재계획** — 위협장을 비용으로 A* 재실행, 채점기가 "이상"이라 보는 경로 확인
3. object 심기 + `los_surface` 분리 → cover(defilade) 부활
4. (선택) exposure hazard 리팩터 / T_EXPOSE 스윕
5. 시간 항·정규화 복원 → mission 가중치 핸들화
6. `tanknav/tune.py` — CMA-ES/베이지안으로 비용맵 가중치 탐색 (최종 목표)

---

## 9. 기존 재사용 자산 (이미 구현됨)

- `tanknav/`: config, mapio(MapBundle 로드), mapping(보정+base cost), risk(Enemy,
  ridge/open/los_risk, **is_visible**, parse_enemies), planning(A*), viz, perception(스텁).
- 데이터 레이어(.npy, fin/ 루트): heightmap / heightmap_filled / slope / cost(base v3) /
  obstacle / ridge·open·los_risk. 최신 ts = `20260601_105333`.
- K2 제원: 전고 2.4m, 전폭 3.6m.
