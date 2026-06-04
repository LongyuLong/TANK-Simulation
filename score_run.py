"""
채점 1회 실행 = run 폴더 1개 (자기완결 기록).
  python score_run.py [label]

fin/runs/<ts>[_label]/ 에 params·enemies·map_info·paths·scores·per_segment·
threat_*.npy·threatmap.png 저장. 적 배치는 build_enemies() 수정.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import glob

from tanknav import mapio, risk, eval as ev, config, runlog, mapping


def build_enemies():
    return [
        risk.Enemy("tank",     [(40, 18)]),
        risk.Enemy("infantry", [(20, 42)]),
        risk.Enemy("patrol",   risk.bresenham(48, 8, 48, 52)),
    ]


def get_bundle_field(bundle, candidates):
    """
    MapBundle 안의 필드명이 프로젝트 버전에 따라 다를 수 있어서
    여러 후보 이름 중 존재하는 값을 찾아 반환.
    """
    for name in candidates:
        if hasattr(bundle, name):
            return name, getattr(bundle, name)
    return None, None


def set_bundle_field(bundle, field_name, value):
    if field_name:
        setattr(bundle, field_name, value)


def apply_virtual_rocks_to_bundle(bundle):
    """
    config.py에 설정한 VIRTUAL_ROCKS를 bundle의 cost / obstacle에 반영한다.

    전제:
    - tanknav/mapping.py 안에 apply_virtual_rocks() 함수가 추가되어 있어야 함.
    - config.py 안에 ENABLE_VIRTUAL_ROCKS, VIRTUAL_ROCKS 설정이 있어야 함.
    """
    if not getattr(config, "ENABLE_VIRTUAL_ROCKS", False):
        print("[rocks] disabled")
        return bundle

    if not hasattr(mapping, "apply_virtual_rocks"):
        print("[rocks] mapping.apply_virtual_rocks() 없음 → rock 적용 건너뜀")
        return bundle

    cost_name, cost_map = get_bundle_field(
        bundle,
        ["cost_map", "cost", "costmap"]
    )

    obstacle_name, obstacle_map = get_bundle_field(
        bundle,
        ["obstacle", "obstacles", "obstacle_map", "obs"]
    )

    height_name, heightmap = get_bundle_field(
        bundle,
        ["heightmap", "height_map", "height", "hm"]
    )

    if cost_map is None:
        print("[rocks] cost map 필드를 못 찾음 → rock 적용 실패")
        print("[rocks] bundle fields:", dir(bundle))
        return bundle

    if obstacle_map is None:
        print("[rocks] obstacle map 필드를 못 찾음 → rock 적용 실패")
        print("[rocks] bundle fields:", dir(bundle))
        return bundle

    result = mapping.apply_virtual_rocks(
        cost_map,
        obstacle_map,
        heightmap,
    )

    # mapping.apply_virtual_rocks 반환값:
    # cost_map, obstacle_map, rock_mask, los_surface
    new_cost, new_obstacle, rock_mask, los_surface = result

    set_bundle_field(bundle, cost_name, new_cost)
    set_bundle_field(bundle, obstacle_name, new_obstacle)

    # los_surface는 아직 eval/risk가 직접 쓰지 않더라도 bundle에 붙여둠.
    if los_surface is not None:
        setattr(bundle, "los_surface", los_surface)

    if rock_mask is not None:
        rock_cells = int(rock_mask.sum())
    else:
        rock_cells = 0

    print(f"[rocks] enabled: {len(getattr(config, 'VIRTUAL_ROCKS', []))} rocks")
    print(f"[rocks] rock cells: {rock_cells}")
    print(f"[rocks] updated fields: {cost_name}, {obstacle_name}")

    return bundle


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    run_dir = runlog.new_run(label)

    bundle = mapio.load_maps()

    # ================================
    # Virtual Rock 적용 구간
    # ================================
    # Yakis.py로 만든 원본 맵 데이터를 불러온 뒤,
    # config.py에 적어둔 큰돌 정보를 cost/obstacle map에 반영한다.
    # 이후 full_run에서 planning/eval/viz가 이 변경된 bundle을 사용한다.
    bundle = apply_virtual_rocks_to_bundle(bundle)

    enemies = build_enemies()

    path_files = sorted(glob.glob(str(config.DATA_DIR / "path_*.npy")))[-4:]
    named = {
        p.replace("\\", "/").split("/")[-1]: ev.load_path_cells(p)
        for p in path_files
    }

    precomp, _ = runlog.full_run(run_dir, bundle, enemies, named)

    nz = int((precomp.intensity > 0).sum())
    print(f"[run] {run_dir}")
    print(
        f"  맵 {bundle.ts}  적 {len(enemies)}  경로 {len(named)}  "
        f"위협>0 셀 {nz}/{precomp.intensity.size}"
    )
    print(
        f"  저장: params/enemies/map_info/paths.json, scores.(csv|txt), "
        f"per_segment.csv, threat_*.npy, threatmap.png"
    )
    print()
    print((run_dir / "scores.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()