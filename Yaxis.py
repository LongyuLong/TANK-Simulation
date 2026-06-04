from flask import Flask, request, jsonify
import numpy as np
import math, csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

app = Flask(__name__)

# ── 설정 ───────────────────────────────────────────────
MAP_SIZE           = 300
GRID_RES           = 5
GRID_N             = MAP_SIZE // GRID_RES + 1
ARRIVE_THR         = 5.0
TURN_THR           = 5.0
CORR_THR           = 3.0
BRAKE_STEPS        = 3
AUTO_SAVE_INTERVAL = 100
STUCK_SPEED        = 0.3
STUCK_STEPS        = 4
MAX_ESCAPE         = 3

# 탈출 시퀀스: (moveWS, moveAD, 지속스텝)
ESCAPE_SEQ = [
    ("S",  "",  3),
    ("S",  "D", 2),
    ("W",  "",  2),
]
ESCAPE_TOTAL = sum(d for _, _, d in ESCAPE_SEQ)

# ── 데이터 ─────────────────────────────────────────────
heightmap    = np.full((GRID_N, GRID_N), np.nan)
obstacle_map = np.zeros((GRID_N, GRID_N), dtype=bool)
raw_log      = []

# ── 전차 상태 ──────────────────────────────────────────
state = {
    "x": 0.0, "y": 0.0, "z": 0.0,
    "heading": 0.0,
    "api_yaw": 0.0,
    "px": None, "pz": None,
    "speed":   0.0,
}

# ── FSM ────────────────────────────────────────────────
fsm = {
    "phase":        "MOVE_Z",
    "pos_z":        True,
    "target_x":     GRID_RES,
    "turn_target":  0.0,
    "brake_next":   "",
    "brake_cnt":    0,
    "stuck_cnt":    0,
    "escape_cnt":   0,
    "escape_step":  0,
    "prev_phase":   "MOVE_Z",   # 탈출 후 복귀할 페이즈
}

# ── 유틸 ───────────────────────────────────────────────
def norm_angle(a):
    return (a + 180) % 360 - 180

def update_heading(x, z):
    if state["px"] is not None:
        dx, dz = x - state["px"], z - state["pz"]
        dist   = math.sqrt(dx**2 + dz**2)
        state["speed"] = dist
        if dist > 0.3:
            state["heading"] = math.degrees(math.atan2(dx, dz))
        else:
            state["heading"] = state["api_yaw"]
    else:
        state["speed"] = 0.0
    state["px"], state["pz"] = state["x"], state["z"]

def record(x, y, z):
    gi = int(round(x / GRID_RES))
    gk = int(round(z / GRID_RES))
    if 0 <= gi < GRID_N and 0 <= gk < GRID_N:
        prev = heightmap[gi, gk]
        heightmap[gi, gk] = y if np.isnan(prev) else (prev + y) / 2.0
    raw_log.append((round(x,2), round(y,4), round(z,2)))

    if len(raw_log) % AUTO_SAVE_INTERVAL == 0:
        np.save("heightmap_autosave.npy", heightmap)
        np.save("obstacle_autosave.npy",  obstacle_map)
        covered = int((~np.isnan(heightmap)).sum())
        print(f"💾 자동저장 ({len(raw_log)}스텝 | 커버 {covered}/{GRID_N*GRID_N})")

# ── stuck 감지 ─────────────────────────────────────────
def check_stuck():
    """
    이동 중 페이즈(MOVE_Z, MOVE_X)에서만 호출.
    True 반환 시 ESCAPE로 전환됨.
    """
    if state["speed"] < STUCK_SPEED:
        fsm["stuck_cnt"] += 1
    else:
        fsm["stuck_cnt"] = 0

    if fsm["stuck_cnt"] < STUCK_STEPS:
        return False

    # stuck 판정
    fsm["stuck_cnt"] = 0
    fsm["escape_cnt"] += 1

    if fsm["escape_cnt"] > MAX_ESCAPE:
        # 포기 → obstacle 마킹 후 스킵
        gi = int(round(state["x"] / GRID_RES))
        gk = int(round(state["z"] / GRID_RES))
        if 0 <= gi < GRID_N and 0 <= gk < GRID_N:
            obstacle_map[gi, gk] = True
        print(f"⛔ stuck 포기 → obstacle 마킹 "
              f"grid({gi},{gk}) / world({state['x']:.1f},{state['z']:.1f})")
        fsm["escape_cnt"] = 0
        # 현재 페이즈 유지하고 진행 (보간이 채워줌)
        return False

    fsm["prev_phase"]  = fsm["phase"]
    fsm["phase"]       = "ESCAPE"
    fsm["escape_step"] = 0
    print(f"🔄 stuck! 탈출 시도 {fsm['escape_cnt']}/{MAX_ESCAPE} "
          f"(spd={state['speed']:.2f})")
    return True

# ── 탈출 명령 ──────────────────────────────────────────
def escape_cmd(step):
    elapsed = 0
    for ws, ad, duration in ESCAPE_SEQ:
        elapsed += duration
        if step < elapsed:
            return {
                "moveWS":   {"command": ws, "weight": 0.6},
                "moveAD":   {"command": ad, "weight": 0.8},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False
            }
    return fwd_cmd()

# ── 명령 헬퍼 ──────────────────────────────────────────
def fwd_cmd(turn="", turn_w=0.0):
    return {
        "moveWS":   {"command": "W",  "weight": 0.25},
        "moveAD":   {"command": turn, "weight": turn_w},
        "turretQE": {"command": "",   "weight": 0.0},
        "turretRF": {"command": "",   "weight": 0.0},
        "fire": False
    }

def stop_cmd():
    return {
        "moveWS":   {"command": "STOP", "weight": 1.0},
        "moveAD":   {"command": "",     "weight": 0.0},
        "turretQE": {"command": "",     "weight": 0.0},
        "turretRF": {"command": "",     "weight": 0.0},
        "fire": False
    }

def turn_cmd(target_yaw):
    diff   = norm_angle(target_yaw - state["heading"])
    side   = "D" if diff > 0 else "A"
    turn_w = max(min(abs(diff) / 45.0, 1.0), 0.5)
    return {
        "moveWS":   {"command": "STOP", "weight": 1.0},
        "moveAD":   {"command": side,   "weight": turn_w},
        "turretQE": {"command": "",     "weight": 0.0},
        "turretRF": {"command": "",     "weight": 0.0},
        "fire": False
    }

def get_corr(target_h):
    diff = norm_angle(target_h - state["heading"])
    if abs(diff) > CORR_THR:
        return ("D" if diff > 0 else "A"), min(abs(diff) / 30.0, 0.6)
    return "", 0.0

# ── FSM 전이 ───────────────────────────────────────────
def fsm_tick():
    x, z = state["x"], state["z"]
    h    = state["heading"]
    ph   = fsm["phase"]

    if ph == "MOVE_Z":
        if check_stuck():
            return
        at_end = (fsm["pos_z"]     and z >= MAP_SIZE - ARRIVE_THR) or \
                 (not fsm["pos_z"] and z <= ARRIVE_THR)
        if at_end:
            if x >= MAP_SIZE - GRID_RES + 1:
                fsm["phase"] = "DONE"
                save()
                print("🏁 완료!")
            else:
                fsm.update({"phase":"BRAKE","brake_next":"TURN_X",
                            "brake_cnt":0,"stuck_cnt":0})
                print(f"▶ BRAKE→TURN_X (x={x:.1f} z={z:.1f})")

    elif ph == "BRAKE":
        fsm["brake_cnt"] += 1
        if fsm["brake_cnt"] >= BRAKE_STEPS:
            nxt = fsm["brake_next"]
            if nxt == "TURN_X":
                fsm.update({"phase":"TURN_X","turn_target":90.0})
                print(f"▶ TURN_X  (h={h:.1f}°)")
            elif nxt == "TURN_Z":
                tgt = 0.0 if fsm["pos_z"] else 180.0
                fsm.update({"phase":"TURN_Z","turn_target":tgt})
                print(f"▶ TURN_Z  목표={tgt:.0f}° (h={h:.1f}°)")

    elif ph == "TURN_X":
        if abs(norm_angle(h - 90.0)) < TURN_THR:
            fsm.update({"phase":"MOVE_X","target_x": x + GRID_RES,
                        "stuck_cnt":0,"escape_cnt":0})
            print(f"▶ MOVE_X  x={x:.1f} → {x+GRID_RES:.1f}")

    elif ph == "MOVE_X":
        if check_stuck():
            return
        if x >= fsm["target_x"] - 1.0:
            fsm["pos_z"] = not fsm["pos_z"]
            fsm.update({"phase":"BRAKE","brake_next":"TURN_Z",
                        "brake_cnt":0,"stuck_cnt":0})
            print(f"▶ BRAKE→TURN_Z")

    elif ph == "TURN_Z":
        if abs(norm_angle(h - fsm["turn_target"])) < TURN_THR:
            fsm.update({"phase":"MOVE_Z","stuck_cnt":0,"escape_cnt":0})
            print(f"▶ MOVE_Z  ({'↑' if fsm['pos_z'] else '↓'}Z x={x:.1f})")

    elif ph == "ESCAPE":
        fsm["escape_step"] += 1
        if fsm["escape_step"] >= ESCAPE_TOTAL:
            fsm["phase"] = fsm["prev_phase"]
            print(f"✅ 탈출 완료 → {fsm['prev_phase']} 재개")

# ── FSM 명령 생성 ──────────────────────────────────────
def fsm_cmd():
    ph = fsm["phase"]
    if ph == "DONE":
        return stop_cmd()
    elif ph == "MOVE_Z":
        t, w = get_corr(0.0 if fsm["pos_z"] else 180.0)
        return fwd_cmd(t, w)
    elif ph == "BRAKE":
        return stop_cmd()
    elif ph in ("TURN_X", "TURN_Z"):
        return turn_cmd(fsm["turn_target"])
    elif ph == "MOVE_X":
        t, w = get_corr(90.0)
        return fwd_cmd(t, w)
    elif ph == "ESCAPE":
        return escape_cmd(fsm["escape_step"])
    return stop_cmd()

# ── 저장 ───────────────────────────────────────────────
def make_slope_and_cost(hm):
    grad_x, grad_z = np.gradient(hm, GRID_RES)
    slope_deg      = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_z**2)))
    cost_map                  = np.ones_like(slope_deg)
    cost_map[slope_deg > 15]  = 3.0
    cost_map[slope_deg > 30]  = 10.0
    cost_map[slope_deg > 45]  = np.inf
    cost_map[obstacle_map]    = np.inf
    return slope_deg, cost_map

def save_image(hm_filled, slope_deg, cost_map, ts):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im0 = axes[0].imshow(hm_filled, origin="lower", cmap="terrain", aspect="equal")
    axes[0].set_title("HeightMap (Y)")
    axes[0].set_xlabel("Z축"); axes[0].set_ylabel("X축")
    plt.colorbar(im0, ax=axes[0], label="고도 (m)")

    im1 = axes[1].imshow(slope_deg, origin="lower", cmap="hot_r",
                         aspect="equal", vmin=0, vmax=45)
    axes[1].set_title("Slope Map (도)")
    axes[1].set_xlabel("Z축"); axes[1].set_ylabel("X축")
    plt.colorbar(im1, ax=axes[1], label="경사도 (°)")

    cost_vis = np.where(np.isinf(cost_map), 20, cost_map)
    im2 = axes[2].imshow(cost_vis, origin="lower", cmap="RdYlGn_r", aspect="equal")
    axes[2].set_title("Cost Map (A* 입력)")
    axes[2].set_xlabel("Z축"); axes[2].set_ylabel("X축")
    plt.colorbar(im2, ax=axes[2], label="비용")

    obs_vis = np.ma.masked_where(~obstacle_map, obstacle_map.astype(float))
    for ax in axes:
        ax.imshow(obs_vis, origin="lower", cmap="cool", aspect="equal", alpha=0.8)

    plt.suptitle(f"Tank Challenge HeightMap — {ts}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"heightmap_{ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🖼️  heightmap_{ts}.png 저장")

def save():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    np.save(f"heightmap_{ts}.npy",    heightmap)
    np.save(f"obstacle_{ts}.npy",     obstacle_map)
    with open(f"heightmap_raw_{ts}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["x","y","z"]] + raw_log)

    # 보간
    rows, cols = np.indices((GRID_N, GRID_N))
    known      = ~np.isnan(heightmap)
    pts        = np.column_stack([rows[known], cols[known]])
    vals       = heightmap[known]
    all_pts    = np.column_stack([rows.ravel(), cols.ravel()])

    filled = griddata(pts, vals, all_pts, method='linear').reshape(GRID_N, GRID_N)
    nan_mask = np.isnan(filled)
    if nan_mask.any():
        nn = griddata(pts, vals, all_pts, method='nearest').reshape(GRID_N, GRID_N)
        filled[nan_mask] = nn[nan_mask]
    np.save(f"heightmap_{ts}_filled.npy", filled)

    slope_deg, cost_map = make_slope_and_cost(filled)
    np.save(f"slope_{ts}.npy",    slope_deg)
    np.save(f"cost_map_{ts}.npy", cost_map)

    save_image(filled, slope_deg, cost_map, ts)

    covered = int(known.sum())
    total   = GRID_N * GRID_N
    print(f"\n{'='*55}")
    print(f"💾 저장 완료: {ts}")
    print(f"   실측 커버리지 : {covered}/{total} ({covered/total*100:.1f}%)")
    print(f"   보간 후        : {total}/{total} (100%)")
    print(f"   Y 범위         : {np.nanmin(heightmap):.2f} ~ {np.nanmax(heightmap):.2f}")
    print(f"   장애물 셀      : {int(obstacle_map.sum())}개")
    print(f"   파일           : heightmap_{ts}.npy / _filled.npy")
    print(f"                    slope_{ts}.npy / cost_map_{ts}.npy")
    print(f"                    obstacle_{ts}.npy / .png")
    print(f"{'='*55}")

# ── 엔드포인트 ─────────────────────────────────────────
@app.route('/init', methods=['GET'])
def init():
    global heightmap, obstacle_map, raw_log
    heightmap    = np.full((GRID_N, GRID_N), np.nan)
    obstacle_map = np.zeros((GRID_N, GRID_N), dtype=bool)
    raw_log      = []
    fsm.update({
        "phase":"MOVE_Z","pos_z":True,"target_x":GRID_RES,
        "turn_target":0.0,"brake_next":"","brake_cnt":0,
        "stuck_cnt":0,"escape_cnt":0,"escape_step":0,"prev_phase":"MOVE_Z",
    })
    state.update({
        "x":0.0,"y":0.0,"z":0.0,"heading":0.0,"api_yaw":0.0,
        "px":None,"pz":None,"speed":0.0,
    })
    print("🚀 탐색 시작")
    return jsonify({
        "startMode":             "start",
        "blStartX":              2,
        "blStartY":              10,
        "blStartZ":              2,
        "rdStartX":              280,
        "rdStartY":              10,
        "rdStartZ":              280,
        "trackingMode":          True,
        "detectMode":            False,
        "logMode":               True,
        "stereoCameraMode":      False,
        "enemyTracking":         False,
        "saveSnapshot":          False,
        "saveLog":               False,
        "saveLidarData":         False,
        "lux":                   30000,
        "destoryObstaclesOnHit": True,
    })

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data: return jsonify({"error":"No data"}), 400

    pos = data.get("playerPos", {})
    x   = pos.get("x", state["x"])
    z   = pos.get("z", state["z"])

    state["api_yaw"] = data.get("playerBodyX", state["api_yaw"])
    update_heading(x, z)
    state["x"] = x
    state["y"] = pos.get("y", state["y"])
    state["z"] = z

    record(state["x"], state["y"], state["z"])

    covered = int((~np.isnan(heightmap)).sum())
    print(f"[{fsm['phase']:8s}] "
          f"({state['x']:6.1f},{state['y']:5.2f},{state['z']:6.1f}) "
          f"h={state['heading']:6.1f}° yaw={state['api_yaw']:6.1f}° "
          f"spd={state['speed']:4.1f} sc={fsm['stuck_cnt']} "
          f"| {covered}/{GRID_N*GRID_N}")
    return jsonify({"status":"success","control":""})

@app.route('/get_action', methods=['POST'])
def get_action():
    fsm_tick()
    return jsonify(fsm_cmd())

@app.route('/collision', methods=['POST'])
def collision():
    data = request.get_json()
    if data:
        pos = data.get('position', {})
        x, z = pos.get('x', 0), pos.get('z', 0)
        gi = int(round(x / GRID_RES))
        gk = int(round(z / GRID_RES))
        if 0 <= gi < GRID_N and 0 <= gk < GRID_N:
            obstacle_map[gi, gk] = True
            print(f"🚧 충돌 장애물: grid({gi},{gk}) "
                  f"world({x:.1f},{z:.1f}) | 총 {int(obstacle_map.sum())}개")
    return jsonify({'status': 'success'})

@app.route('/detect',          methods=['POST'])
def detect():          return jsonify([])
@app.route('/stereo_image',    methods=['POST'])
def stereo_image():    return jsonify({"result":"success"})
@app.route('/update_bullet',   methods=['POST'])
def update_bullet():   return jsonify({"status":"OK"})
@app.route('/set_destination', methods=['POST'])
def set_destination(): return jsonify({"status":"OK"})
@app.route('/update_obstacle', methods=['POST'])
def update_obstacle(): return jsonify({"status":"success"})
@app.route('/start',           methods=['GET'])
def start():           return jsonify({"control":""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)