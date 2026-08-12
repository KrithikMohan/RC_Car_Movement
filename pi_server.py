import os
import time
import math
import shutil
import threading
import cv2
from flask import Flask, Response, request, jsonify, send_from_directory
from ugv_command import send_command

app = Flask(__name__)

slow_speed=0.15
turn_speed=0.10
high_speed=0.30

# Motor Command Mapping
COMMAND_MAP = {
    'low': {
        'f': f'{{"T":1,"L":{slow_speed},"R":{slow_speed}}}',
        'b': f'{{"T":1,"L":{-slow_speed},"R":{-slow_speed}}}',
        'l': f'{{"T":1,"L":{-turn_speed},"R":{turn_speed}}}',
        'r': f'{{"T":1,"L":{turn_speed},"R":{-turn_speed}}}',
        's': '{"T":1,"L":0,"R":0}'
    },
    'high': {
        'f': f'{{"T":1,"L":{high_speed},"R":{high_speed}}}',
        'b': f'{{"T":1,"L":{-high_speed},"R":{-high_speed}}}',
        'l': f'{{"T":1,"L":{-turn_speed},"R":{turn_speed}}}',
        'r': f'{{"T":1,"L":{turn_speed},"R":{-turn_speed}}}',
        's': '{"T":1,"L":0,"R":0}'
    }
}

# State Variables
current_speed_mode = 'low'
last_cmd_time = time.time()
current_cmd = 's'

# Telemetry and Odometry State
telemetry_data = {
    'pos_x': 0.0,
    'pos_y': 0.0,
    'distance_home_m': 0.0,
    'heading_home_deg': 0.0,
    'speed_mode': 'low',
    'battery_pct': 100,
    'voltage': 12.6,
    'is_recording': False,
    'rec_file': '',
    'free_storage_gb': 0.0
}

# Video Recording Configuration
RECORD_DIR = os.path.expanduser('~/videos')
os.makedirs(RECORD_DIR, exist_ok=True)
video_writer = None
camera = cv2.VideoCapture(0)


def send_ugv_command(cmd_str):
    """Send a command to UGV02 via ugv_command module."""
    try:
        response = send_command(cmd_str)
        if response:
            print(f"[UGV02 RESPONSE] {response}")
        return response
    except Exception as e:
        print(f"[UGV02 ERROR] {e}")
        return None


def watchdog_thread():
    """Failsafe watchdog to stop motors if network connection drops."""
    global current_cmd
    while True:
        if time.time() - last_cmd_time > 0.5 and current_cmd != 's':
            current_cmd = 's'
            send_ugv_command(COMMAND_MAP[current_speed_mode]['s'])
        time.sleep(0.1)


threading.Thread(target=watchdog_thread, daemon=True).start()


def update_telemetry_loop():
    """Background worker updating system storage, power, and positioning."""
    global telemetry_data
    while True:
        total, used, free = shutil.disk_usage("/")
        telemetry_data['free_storage_gb'] = round(free / (1024 ** 3), 2)
        telemetry_data['speed_mode'] = current_speed_mode

        # Simulated dead reckoning from wheel command direction
        if current_cmd == 'f':
            telemetry_data['pos_y'] += 0.05
        elif current_cmd == 'b':
            telemetry_data['pos_y'] -= 0.05
        elif current_cmd == 'l':
            telemetry_data['pos_x'] -= 0.03
        elif current_cmd == 'r':
            telemetry_data['pos_x'] += 0.03

        x = telemetry_data['pos_x']
        y = telemetry_data['pos_y']
        telemetry_data['distance_home_m'] = round(math.sqrt(x**2 + y**2), 2)
        heading_rad = math.atan2(-x, -y)
        telemetry_data['heading_home_deg'] = round(math.degrees(heading_rad) % 360, 1)

        time.sleep(0.2)


threading.Thread(target=update_telemetry_loop, daemon=True).start()


def gen_frames():
    global video_writer
    while True:
        success, frame = camera.read()
        if not success:
            break

        if telemetry_data['is_recording'] and video_writer is not None:
            video_writer.write(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/control', methods=['POST'])
def control():
    global last_cmd_time, current_cmd
    data = request.json or {}
    cmd = data.get('command', 's')
    last_cmd_time = time.time()
    current_cmd = cmd

    if cmd in COMMAND_MAP[current_speed_mode]:
        send_ugv_command(COMMAND_MAP[current_speed_mode][cmd])
    return jsonify({'status': 'ok', 'command': cmd})


@app.route('/telemetry', methods=['GET'])
def telemetry():
    return jsonify(telemetry_data)


@app.route('/speed', methods=['POST'])
def set_speed():
    global current_speed_mode
    data = request.json or {}
    mode = data.get('mode', 'low')
    if mode in COMMAND_MAP:
        current_speed_mode = mode
    return jsonify({'status': 'ok', 'speed_mode': current_speed_mode})


@app.route('/recording', methods=['POST'])
def handle_recording():
    global video_writer
    data = request.json or {}
    action = data.get('action', 'stop')

    if action == 'start' and not telemetry_data['is_recording']:
        filename = f"rec_{int(time.time())}.mp4"
        filepath = os.path.join(RECORD_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))
        telemetry_data['is_recording'] = True
        telemetry_data['rec_file'] = filename
    elif action == 'stop' and telemetry_data['is_recording']:
        telemetry_data['is_recording'] = False
        if video_writer:
            video_writer.release()
            video_writer = None

    return jsonify({
        'status': 'ok',
        'is_recording': telemetry_data['is_recording'],
        'rec_file': telemetry_data['rec_file']
    })


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(RECORD_DIR, filename, as_attachment=True)


@app.route('/reset_origin', methods=['POST'])
def reset_origin():
    telemetry_data['pos_x'] = 0.0
    telemetry_data['pos_y'] = 0.0
    telemetry_data['distance_home_m'] = 0.0
    telemetry_data['heading_home_deg'] = 0.0
    return jsonify({'status': 'ok', 'message': 'Origin reset to 0,0'})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    send_ugv_command(COMMAND_MAP[current_speed_mode]['s'])

    def _shutdown():
        time.sleep(1)
        os.system('sudo shutdown -h now')

    threading.Thread(target=_shutdown).start()
    return jsonify({'status': 'ok', 'message': 'Shutting down Raspberry Pi...'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
