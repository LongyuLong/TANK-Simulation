# OpenMP 중복 런타임(libiomp5md.dll) 충돌 회피 — torch/numpy import '전에' 설정 필수.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from flask import Flask, request, jsonify, Response, send_file
import torch
import json
import queue
import threading
import math
from ultralytics import YOLO

# tanknav 패키지(프로젝트 루트) import 경로 추가
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pursuit
# [교체] 평면 A*(path_planner) → 우리 글로벌 플래너(Option S) 어댑터
from tanknav.sim_adapter import plan_path, START_X, START_Z
from tanknav import sim_adapter
from map_parser import parse_map

ENGAGE_POLICY = "balanced"   # 교전 결심 정책 (stub) — 추후 web 운용자 승인으로 교체

# ============================================================
# Flask 앱 초기화
# ============================================================
import os

app   = Flask(__name__)
model = YOLO('yolov8n.pt')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

current_heading   = 0.0
info_call_count   = 0
dynamic_obstacles = []
goal              = {'x': 142.47, 'z': 214.1}
current_pos       = {'x': START_X, 'z': START_Z}   # 최신 전차 위치 (SA 재계획용)

# 정적 장애물: 오브젝트를 설치·저장한 맵에서 로드 (없으면 빈 맵 = 지형만).
#   동적(차/사람)은 sim_adapter가 자동 제외 → 런타임 인식으로 처리.
#   경로는 환경변수 TANK_MAP_FILE 로 덮어쓸 수 있음(이식성). 없으면 아래 기본값.
MAP_FILE = os.environ.get(
    "TANK_MAP_FILE", r"C:\Users\acorn\Documents\Tank Challenge\map\Test1.map")
obstacle_list = []
if MAP_FILE:
    try:
        obstacle_list = parse_map(MAP_FILE)
        print(f"🗺️  맵 로드: {MAP_FILE} ({len(obstacle_list)}개 정적 장애물)")
    except FileNotFoundError:
        print(f"⚠️  맵 파일 없음 ({MAP_FILE}) — 빈 맵으로 시작")
    except Exception as e:
        print(f"⚠️  맵 파싱 실패 ({MAP_FILE}): {e} — 빈 맵으로 시작")
else:
    print("🗺️  빈 맵으로 시작 (정적 장애물 없음)")

# 정적 오브젝트를 글로벌 플래너에 등록 (config.OBJECT_TYPES 활성 타입만 cost+risk 반영)
sim_adapter.set_static_obstacles(obstacle_list)

print("⏸️  경로 미설정 - 브라우저 localhost:5000 에서 목표를 설정하세요")

# SSE 브로드캐스트 큐 (최대 10개)
sse_clients = []
sse_lock    = threading.Lock()


# ── SSE 브로드캐스트 ──────────────────────────────────────────
def broadcast(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)


# ============================================================
# 대시보드 서빙
# ============================================================
@app.route('/')
def dashboard():
    return send_file(os.path.join(BASE_DIR, 'dashboard.html'))


@app.route('/layers')
def layers():
    """대시보드 레이어 그리드(heightmap/threat/risk…) JSON. 로드 시 + 계획/탐지 후 호출."""
    try:
        return jsonify(sim_adapter.layers())
    except Exception as e:
        print(f"⚠️ /layers 실패: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stream')
def stream():
    q = queue.Queue(maxsize=50)
    with sse_lock:
        sse_clients.append(q)

    def generate():
        # 현재 상태 즉시 전송
        try:
            q.put_nowait(f"data: {json.dumps(_full_state())}\n\n")
        except Exception:
            pass
        while True:
            try:
                msg = q.get(timeout=20)
                yield msg
            except queue.Empty:
                # heartbeat - 연결 유지
                yield ": heartbeat\n\n"
            except GeneratorExit:
                break
            except Exception:
                break
        with sse_lock:
            if q in sse_clients:
                sse_clients.remove(q)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


def _full_state():
    """현재 전체 상태 dict"""
    all_obs = [dict(o, dynamic=False) for o in obstacle_list] + \
              [dict(o, dynamic=True)  for o in dynamic_obstacles]
    return {
        'tank':      {'x': 0, 'z': 0, 'heading': current_heading},
        'path':      pursuit.waypoints,
        'obstacles': all_obs,
        'wp_index':  pursuit.current_wp_index,
        'dist':      0,
        'err':       0,
    }


# ============================================================
# 경로 계획 API
# ============================================================
@app.route('/plan_path', methods=['POST'])
def api_plan_path():
    global goal
    data = request.get_json(force=True) or {}
    gx = float(data.get('goal_x', goal['x']))
    gz = float(data.get('goal_z', goal['z']))
    goal = {'x': gx, 'z': gz}

    # 정적 오브젝트는 어댑터(set_static_obstacles)가 보유 → 여기선 동적만 전달.
    sx, sz = current_pos['x'], current_pos['z']   # 현재 위치에서 계획
    print(f"🗺️  계획: 출발=({sx:.0f},{sz:.0f}) 목표=({gx},{gz}) "
          f"정적={len(obstacle_list)} 동적={len(dynamic_obstacles)}")
    wps = plan_path(dynamic_obstacles, sx, sz, gx, gz)
    if wps is None:
        return jsonify({'error': 'A* 경로 탐색 실패'}), 500

    pursuit.waypoints = wps
    broadcast({
        'path': wps,
        'log': f'A* 완료: {len(wps)}개 WP (목표 {gx:.1f},{gz:.1f})'
    })
    return jsonify({'waypoints': wps, 'count': len(wps)})


# ============================================================
# 시뮬레이터 엔드포인트
# ============================================================
@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400
    image.save('temp_image.jpg')
    results    = model('temp_image.jpg')
    detections = results[0].boxes.data.cpu().numpy()
    target_classes = {0:"tank",1:"rock",2:"car",7:"truck",15:"rock"}
    filtered = []
    for box in detections:
        cid = int(box[5])
        if cid in target_classes:
            filtered.append({
                'className': target_classes[cid],
                'bbox': [float(c) for c in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00', 'filled': False, 'updateBoxWhileMoving': False
            })
    return jsonify(filtered)


@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    left  = request.files.get('left_image')
    right = request.files.get('right_image')
    if not left or not right:
        return jsonify({"result": "error", "message": "image missing"}), 400
    left.save("temp_left.jpg"); right.save("temp_right.jpg")
    return jsonify({"result": "success"})


@app.route('/info', methods=['POST'])
def info():
    global current_heading, info_call_count
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    info_call_count += 1
    prev    = current_heading
    lidar_y = data.get("lidarRotation", {}).get("y", None)

    if lidar_y is not None:
        current_heading = lidar_y
    else:
        current_heading = data.get("playerBodyY", current_heading)

    if info_call_count <= 5:
        print(f"📡 /info RAW #{info_call_count}: lidarY={lidar_y}")
    elif info_call_count % 50 == 0:
        print(f"📡 /info #{info_call_count} | {prev:.1f}° → {current_heading:.1f}°")

    return jsonify({"status": "success", "control": ""})


@app.route('/get_action', methods=['POST'])
def get_action():
    global current_heading

    data  = request.get_json(force=True)
    pos   = data.get("position", {})
    pos_x = pos.get("x", 0)
    pos_z = pos.get("z", 0)
    current_pos['x'], current_pos['z'] = pos_x, pos_z   # SA 재계획용 최신 위치

    move_ws, move_ad, pursuit.current_wp_index, move_weight, turn_weight = pursuit.pure_pursuit(
        pos_x, pos_z, current_heading, pursuit.current_wp_index
    )

    # 현재 WP까지 거리 / 오차 계산 (시각화용)
    dist, err = 0.0, 0.0
    if pursuit.waypoints and pursuit.current_wp_index < len(pursuit.waypoints):
        wp   = pursuit.waypoints[pursuit.current_wp_index]
        dist = math.hypot(wp['x']-pos_x, wp['z']-pos_z)

    broadcast({
        'tank':     {'x': pos_x, 'z': pos_z, 'heading': current_heading},
        'wp_index': pursuit.current_wp_index,
        'dist':     dist,
    })

    command = {
        "moveWS":   {"command": move_ws, "weight": move_weight},
        "moveAD":   {"command": move_ad, "weight": turn_weight},
        "turretQE": {"command": "",      "weight": 0.0},
        "turretRF": {"command": "",      "weight": 0.0},
        "fire": False
    }
    print(f"🔁 CMD: WS={move_ws}({move_weight:.2f}) AD={move_ad}({turn_weight:.2f})")
    print("---")
    return jsonify(command)


@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global dynamic_obstacles
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error'}), 400
    obs = data.get('obstacles', [])
    if not obs:
        return jsonify({'status': 'success'})

    # 정적 장애물 bbox와 비교해서 새로운 것만 동적으로 처리
    static_keys = {(round(o['x_min'], 1), round(o['z_min'], 1)) for o in obstacle_list}
    new_dyn = [o for o in obs
               if (round(o['x_min'], 1), round(o['z_min'], 1)) not in static_keys]
    if not new_dyn:
        return jsonify({'status': 'success'})

    dynamic_obstacles = new_dyn

    # ── SA 루프: 탐지 → known-set 갱신 → (결심) → 재계획 ──────────────
    def _name(o):
        return str(o.get('className') or o.get('name') or o.get('type') or '').lower()
    enemies_seen = [o for o in new_dyn if 'tank' in _name(o)]   # 적전차 = 위협
    blockers     = [o for o in new_dyn if 'tank' not in _name(o)]  # 차/사람 등 = 회피

    cxz  = (current_pos['x'], current_pos['z'])
    gxz  = (goal['x'], goal['z'])
    logs = []

    # 적전차 → known-set 추가 + 교전 결심 근거
    for e in enemies_seen:
        ex = (e['x_min'] + e['x_max']) / 2
        ez = (e['z_min'] + e['z_max']) / 2
        a = sim_adapter.assess(cxz, gxz, (ex, ez), 'tank')   # 추가 전 평가
        d = sim_adapter.decide(a, policy=ENGAGE_POLICY)
        sim_adapter.add_enemy(ex, ez, 'tank')                # known-set 반영
        logs.append(f"적전차@({ex:.0f},{ez:.0f}) risk={a['self_risk']} can_engage={a['can_engage']} → {d}")
        print(f"🎯 {logs[-1]}")

    # 재계획: 정적 오브젝트(어댑터 보유) + 회피대상(blocker, 동적) + 갱신 known-set
    wps = sim_adapter.replan(cxz, gxz, obstacles=blockers)
    if wps:
        pursuit.waypoints        = wps
        pursuit.current_wp_index = 0
        logs.append(f"재계획 {len(wps)}wp")
    else:
        logs.append("재계획 실패(경로 없음)")

    broadcast({
        'path': pursuit.waypoints,
        'obstacles': [dict(o, dynamic=False) for o in obstacle_list] +
                     [dict(o, dynamic=True) for o in dynamic_obstacles],
        'log': '; '.join(logs)
    })
    return jsonify({'status': 'success', 'replanned': bool(wps)})


@app.route('/collision', methods=['POST'])
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error'}), 400
    pos = data.get('position', {})
    print(f"💥 Collision: {data.get('objectName')} at ({pos.get('x')}, {pos.get('y')}, {pos.get('z')})")
    return jsonify({'status': 'success'})


@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR"}), 400
    print(f"💥 Bullet: ({data.get('x')},{data.get('y')},{data.get('z')}) hit={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400
    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination: ({x},{y},{z})")
        return jsonify({"status": "OK"})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400


@app.route('/init', methods=['GET'])
def init():
    global current_heading, info_call_count, dynamic_obstacles
    current_heading  = 0.0
    info_call_count  = 0
    dynamic_obstacles = []
    current_pos['x'], current_pos['z'] = START_X, START_Z
    pursuit.reset()
    sim_adapter.clear_enemies()      # known-set 초기화
    print("🛠️ /init 완료 - 경로는 대시보드에서 설정하세요")
    return jsonify({
        "startMode": "start",
        "blStartX": 60, "blStartY": 10, "blStartZ": 27.23,
        "rdStartX": 59, "rdStartY": 10, "rdStartZ": 280,
        "trackingMode": True, "detectMode": False, "logMode": True,
        "stereoCameraMode": False, "enemyTracking": False,
        "saveSnapshot": False, "saveLog": False, "saveLidarData": False,
        "lux": 30000, "destoryObstaclesOnHit": True
    })


@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start received")
    return jsonify({"control": ""})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
