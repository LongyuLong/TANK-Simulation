"""
tanknav 파이프라인 CLI

사용 예:
  python run.py cost  --raw 20260530_142831      # 수집본 → _final 맵 생성
  python run.py risk                             # 지형 리스크만 (적 없음)
  python run.py risk  --enemies                  # 터미널로 적 입력 후 LoS 포함
  python run.py plan  --start 5 5 --goal 55 55   # A* (리스크맵 우선)
  python run.py plan  --start 5 5 --goal 55 55 --cost base
  python run.py all   --start 5 5 --goal 55 55   # risk → plan 연속
  python run.py threat --start 5 5 --goal 55 55  # 위협-인지 계획(Option S), 적 입력

좌표는 그리드 셀 (row col).  실제 m = 셀 × 5
"""
import argparse
import sys

# Windows 콘솔(cp949)에서 이모지/한글 출력 보장
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from tanknav import mapping, risk, planning, mapio


def main():
    ap = argparse.ArgumentParser(description="tanknav 파이프라인")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cost = sub.add_parser("cost", help="수집본 보정 + base cost 생성")
    p_cost.add_argument("--raw", required=True, help="수집 타임스탬프 (예: 20260530_142831)")

    p_risk = sub.add_parser("risk", help="리스크 통합 코스트 생성")
    p_risk.add_argument("--ts", default=None, help="맵 타임스탬프 (기본: 최신)")
    p_risk.add_argument("--enemies", action="store_true", help="터미널 적 입력 + LoS")

    for name in ("plan", "all"):
        pp = sub.add_parser(name, help="A* 경로 계획" + (" (risk 먼저)" if name=="all" else ""))
        pp.add_argument("--start", nargs=2, type=int, required=True, metavar=("ROW","COL"))
        pp.add_argument("--goal",  nargs=2, type=int, required=True, metavar=("ROW","COL"))
        pp.add_argument("--ts", default=None)
        pp.add_argument("--cost", choices=["risk","base"], default="risk")
        if name == "all":
            pp.add_argument("--enemies", action="store_true")

    # 위협-인지 계획 (Option S): 채점기 직결 threat_cost + 비용-인지 Theta*
    p_threat = sub.add_parser("threat", help="위협-인지 글로벌 계획 (Option S)")
    p_threat.add_argument("--start", nargs=2, type=int, required=True, metavar=("ROW","COL"))
    p_threat.add_argument("--goal",  nargs=2, type=int, required=True, metavar=("ROW","COL"))
    p_threat.add_argument("--ts", default=None)
    p_threat.add_argument("--w-surv", type=float, default=None,
                          help="생존 가중(W_SURV_PLAN). 미지정 시 config 기본값")

    args = ap.parse_args()

    if args.cmd == "cost":
        mapping.build_final_maps(args.raw)

    elif args.cmd == "risk":
        bundle = mapio.load_maps(args.ts)
        enemies = risk.parse_enemies(*bundle.shape) if args.enemies else None
        risk.build_risk_cost(bundle, enemies)

    elif args.cmd == "plan":
        planning.plan(tuple(args.start), tuple(args.goal),
                      ts=args.ts, cost_pref=args.cost)

    elif args.cmd == "all":
        bundle = mapio.load_maps(args.ts)
        enemies = risk.parse_enemies(*bundle.shape) if args.enemies else None

        # 중요:
        # risk.build_risk_cost() 안에서 virtual rocks를 먼저 반영한 뒤
        # open risk / combined cost를 새로 만든다.
        # 따라서 바로 이어지는 planning에서는 같은 돌을 중복 적용하지 않는다.
        risk.build_risk_cost(bundle, enemies)
        planning.plan(tuple(args.start), tuple(args.goal),
                      ts=bundle.ts, cost_pref=args.cost, apply_rocks=False)

    elif args.cmd == "threat":
        bundle = mapio.load_maps(args.ts)
        enemies = risk.parse_enemies(*bundle.shape)
        if not enemies:
            print("⚠️  적이 없으면 위협-인지 계획은 무의미. base/risk plan을 쓰세요.")
            return
        planning.plan_threat(tuple(args.start), tuple(args.goal), enemies,
                             ts=args.ts, w_surv=args.w_surv)


if __name__ == "__main__":
    main()
