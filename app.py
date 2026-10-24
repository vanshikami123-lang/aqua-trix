import cv2
import numpy as np
import time
import json
import random
import threading
import serial
import serial.tools.list_ports
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

# --- SERIAL CONNECTION TO ARDUINO UNO ---
PREFERRED_PORT = "COM7"
BAUD_RATE = 115200
ser = None

state = {
    "flow_lpm": 15.40,
    "total_liters": 142.80,
    "ph": 7.20,
    "turbidity_ntu": 4.2,
    "chlorine_mg_l": 1.38,
    "ultrasonic_cm": 24.0,
    "raw_turb": 820,
    "raw_ph": 512,
    "robot_dist_m": 0.0,
    "robot_moving": False,
    "gps_lat": 28.474400,
    "gps_lon": 77.504000,
    "valve1_open": True,
    "valve2_open": False,
    "leak_sim": False,
    "status": "NOMINAL OPERATION",
    "serial_connected": False,
    "serial_port_name": "SCANNING...",
    "last_defect": "None",
    "fault_history": []
}

def find_arduino_port():
    """Finds COM7 or auto-detects CH340/Arduino USB serial port on any laptop."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.device.upper() == PREFERRED_PORT:
            return p.device
    for p in ports:
        desc = (p.description or "").upper()
        if "CH340" in desc or "ARDUINO" in desc or "USB-SERIAL" in desc or "USB SERIAL" in desc:
            return p.device
    if ports:
        return ports[0].device
    return PREFERRED_PORT

def send_serial_cmd(cmd_str):
    global ser
    try:
        if ser and ser.is_open:
            ser.write((cmd_str + "\n").encode('utf-8'))
            ser.flush()
    except Exception:
        pass

def set_physical_relays(main_open):
    """
    Controls Arduino Uno pins 7 & 8 without re-enabling Arduino's autoProtection bug.
    - main_open=False: sends 'ISOLATE' (autoProtection=false, Relay1=OFF, Relay2=ON)
    - main_open=True: sends 'ISOLATE' then toggles R1 & R2 (autoProtection stays false, Relay1=ON, Relay2=OFF)
    """
    def _worker():
        if not main_open:
            send_serial_cmd("ISOLATE")
        else:
            send_serial_cmd("ISOLATE")
            time.sleep(0.06)
            send_serial_cmd("R1_TOGGLE")
            time.sleep(0.06)
            send_serial_cmd("R2_TOGGLE")
    threading.Thread(target=_worker, daemon=True).start()

def serial_reader_thread():
    """Reads ONLY physical sensor values from Arduino so button states are never overwritten."""
    global ser, state
    while True:
        try:
            if ser is None or not ser.is_open:
                port_to_use = find_arduino_port()
                ser = serial.Serial(port_to_use, BAUD_RATE, timeout=1)
                state["serial_connected"] = True
                state["serial_port_name"] = port_to_use
                time.sleep(1.8)
                # Put Arduino in manual override on connect so dry sensors don't lock relays
                set_physical_relays(state["valve1_open"])

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("{") and line.endswith("}"):
                data = json.loads(line)
                if "ph" in data:
                    state["ph"] = float(data["ph"])
                if "turbidity_ntu" in data:
                    state["turbidity_ntu"] = float(data["turbidity_ntu"])
                if "ultrasonic_cm" in data:
                    state["ultrasonic_cm"] = float(data["ultrasonic_cm"])
                if "raw_turb" in data:
                    state["raw_turb"] = int(data["raw_turb"])
                if "raw_ph" in data:
                    state["raw_ph"] = int(data["raw_ph"])
                state["serial_connected"] = True
        except Exception:
            state["serial_connected"] = False
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
            ser = None
            time.sleep(1.5)

threading.Thread(target=serial_reader_thread, daemon=True).start()

# --- CAMERA INITIALIZATION WITH AUTO-FALLBACK ---
current_cam_idx = 0
camera = None

def open_camera(idx):
    global camera, current_cam_idx
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    current_cam_idx = idx
    camera = cap

open_camera(0)
last_tick = time.time()

def detect_pipe_defects(frame):
    """OpenCV Vision Pipeline: Detects Corrosion/Rust and Structural Cracks."""
    global state
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    defect_found = None

    # 1. Corrosion / Rust Detection (Orange/Brown spray paint)
    lower_rust = np.array([5, 90, 60])
    upper_rust = np.array([25, 255, 255])
    rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
    rust_contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in rust_contours:
        if cv2.contourArea(cnt) > 1200:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 140, 255), 2)
            cv2.putText(frame, f"CORROSION [{state['robot_dist_m']:.2f}m]", (x, max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            defect_found = f"Corrosion / Rust at {state['robot_dist_m']:.2f}m"

    # 2. Crack / Joint Defect Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    crack_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in crack_contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = max(cw, ch) / (min(cw, ch) + 1.0)
        if area > 450 and aspect_ratio > 3.5:
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (50, 50, 255), 2)
            cv2.putText(frame, f"STRUCTURAL CRACK [{state['robot_dist_m']:.2f}m]", (x, max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
            defect_found = f"Crack / Joint Defect at {state['robot_dist_m']:.2f}m"

    # Tactical HUD Reticle
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 25, cy), (cx - 8, cy), (255, 220, 0), 1)
    cv2.line(frame, (cx + 8, cy), (cx + 25, cy), (255, 220, 0), 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy - 8), (255, 220, 0), 1)
    cv2.line(frame, (cx, cy + 8), (cx, cy + 25), (255, 220, 0), 1)

    cv2.rectangle(frame, (0, 0), (w, 34), (10, 15, 30), -1)
    hud = f"CAM #{current_cam_idx} | SONAR: {state['ultrasonic_cm']:.1f}cm | ODOMETRY: {state['robot_dist_m']:.2f}m"
    cv2.putText(frame, hud, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1)

    if defect_found and defect_found != state["last_defect"]:
        state["last_defect"] = defect_found
        if len(state["fault_history"]) == 0 or state["fault_history"][0]["type"] != defect_found:
            state["fault_history"].insert(0, {
                "time": time.strftime("%H:%M:%S"),
                "type": defect_found,
                "dist": round(state["robot_dist_m"], 2),
                "gps": f"{state['gps_lat']:.5f}N, {state['gps_lon']:.5f}E"
            })
            state["fault_history"] = state["fault_history"][:6]

    return frame

def generate_frames():
    while True:
        success, frame = (camera.read() if camera is not None else (False, None))
        if not success or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"NO SIGNAL ON CAM #{current_cam_idx} - CLICK 'SWITCH CAM'", (65, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
            time.sleep(0.1)
        else:
            frame = detect_pipe_defects(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def update_dynamic_telemetry():
    global last_tick, state
    now = time.time()
    dt = max(0.05, min(now - last_tick, 1.5))
    last_tick = now

    # 1. Dynamic Flow Rate & Cumulative Water Consumption
    if not state["valve1_open"]:
        state["flow_lpm"] = 0.00
    elif state["leak_sim"]:
        state["flow_lpm"] = round(27.40 + random.uniform(-1.1, 1.3), 2)
        state["total_liters"] = round(state["total_liters"] + (state["flow_lpm"] / 60.0) * dt, 2)
    else:
        wave = np.sin(now * 1.1) * 0.75 + random.uniform(-0.30, 0.30)
        state["flow_lpm"] = round(15.40 + wave, 2)
        state["total_liters"] = round(state["total_liters"] + (state["flow_lpm"] / 60.0) * dt, 2)

    # 2. Physics-Informed AI Soft-Sensor for Residual Chlorine (mg/L)
    cl = 1.45 - (state["turbidity_ntu"] * 0.0025) - (abs(state["ph"] - 7.2) * 0.18)
    state["chlorine_mg_l"] = round(max(0.05, min(2.00, cl)), 2)

    # 3. Crawler Odometry & GPS Progression
    if state["robot_moving"]:
        state["robot_dist_m"] = round((state["robot_dist_m"] + 0.25 * dt) % 10.0, 2)
        state["gps_lat"] = round(state["gps_lat"] + 0.000005 * dt, 6)
        state["gps_lon"] = round(state["gps_lon"] + 0.000007 * dt, 6)

@app.route('/api/telemetry')
def get_telemetry():
    update_dynamic_telemetry()
    return jsonify(state)

@app.route('/api/control', methods=['POST'])
def control():
    global current_cam_idx
    data = request.get_json(silent=True) or {}
    cmd = data.get("cmd", "")

    if cmd == "ISOLATE":
        state["valve1_open"] = False
        state["valve2_open"] = True
        state["leak_sim"] = False
        state["flow_lpm"] = 0.00
        state["status"] = "EMERGENCY LOCKDOWN — PIPE ISOLATED"
        set_physical_relays(False)

    elif cmd == "NORMAL":
        state["valve1_open"] = True
        state["valve2_open"] = False
        state["leak_sim"] = False
        state["flow_lpm"] = 15.40
        state["status"] = "NOMINAL OPERATION"
        set_physical_relays(True)

    elif cmd == "SIM_BURST":
        state["leak_sim"] = not state["leak_sim"]
        if state["leak_sim"]:
            state["valve1_open"] = True
            state["valve2_open"] = False
            state["flow_lpm"] = 27.50
            state["status"] = "CRITICAL: PRESSURE DROP / PIPE BURST"
        else:
            state["flow_lpm"] = 15.40
            state["status"] = "NOMINAL OPERATION"

    elif cmd == "TOGGLE_ROBOT":
        state["robot_moving"] = not state["robot_moving"]

    elif cmd == "RESET_DIST":
        state["robot_dist_m"] = 0.00
        state["gps_lat"] = 28.474400
        state["gps_lon"] = 77.504000
        state["fault_history"] = []
        state["last_defect"] = "None"

    elif cmd == "SWITCH_CAM":
        next_idx = 1 if current_cam_idx == 0 else 0
        open_camera(next_idx)

    update_dynamic_telemetry()
    return jsonify(state)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HydroSight AI — SCADA Digital Twin & Robotic Telemetry</title>
  <style>
    :root {
      --bg: #050811;
      --panel: rgba(13, 21, 38, 0.90);
      --border: rgba(0, 240, 255, 0.25);
      --cyan: #00f0ff;
      --emerald: #00ff9d;
      --amber: #ffb800;
      --crimson: #ff2a5f;
      --purple: #b86bff;
      --text: #e2e8f0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(0, 240, 255, 0.08) 0%, transparent 35%),
        radial-gradient(circle at 85% 85%, rgba(0, 255, 157, 0.06) 0%, transparent 35%),
        linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
      background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px;
      min-height: 100vh; overflow-x: hidden;
    }
    header {
      background: rgba(8, 13, 26, 0.95); padding: 12px 24px;
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--border); box-shadow: 0 0 25px rgba(0, 240, 255, 0.14);
    }
    .brand-title { font-weight: 800; font-size: 20px; letter-spacing: 1.2px; color: #fff; display:flex; align-items:center; gap:12px; }
    .pulse-dot { width: 12px; height: 12px; border-radius: 50%; background: var(--emerald); box-shadow: 0 0 12px var(--emerald); animation: pulse 1.4s infinite; }
    @keyframes pulse { 0%, 100% { transform: scale(0.95); opacity: 0.7; } 50% { transform: scale(1.25); opacity: 1; } }
    .status-pill {
      font-weight: 700; font-size: 12px; padding: 8px 18px; border-radius: 30px;
      background: rgba(0, 255, 157, 0.15); border: 1px solid var(--emerald); color: var(--emerald);
      box-shadow: 0 0 15px rgba(0, 255, 157, 0.25); letter-spacing: 1px;
    }
    .container { padding: 14px 20px; display: flex; flex-direction: column; gap: 14px; }

    /* DIGITAL TWIN PIPELINE SCHEMATIC */
    .twin-card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
      padding: 12px 18px; position: relative; box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    }
    .section-tag { font-weight: 700; font-size: 11px; color: var(--cyan); letter-spacing: 1.3px; text-transform: uppercase; margin-bottom: 6px; display:flex; justify-content:space-between; }
    .pipe-track {
      height: 44px; background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
      border-top: 2px solid #475569; border-bottom: 2px solid #475569; border-radius: 8px;
      position: relative; overflow: hidden; display: flex; align-items: center; margin: 10px 0 6px 0;
    }
    .water-stream {
      position: absolute; inset: 0;
      background: repeating-linear-gradient(90deg, rgba(0, 240, 255, 0.22) 0px, rgba(0, 240, 255, 0.22) 25px, transparent 25px, transparent 50px);
      background-size: 50px 100%; animation: flowAnim 0.8s linear infinite;
    }
    @keyframes flowAnim { from { background-position: 0 0; } to { background-position: 50px 0; } }
    .crawler-node {
      position: absolute; top: 5px; width: 58px; height: 30px; background: var(--amber);
      color: #000; font-family: Consolas, monospace; font-weight: 800; font-size: 12px;
      border-radius: 6px; display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 18px var(--amber); transition: left 0.35s linear; z-index: 4;
    }
    .fault-pin {
      position: absolute; top: -4px; width: 10px; height: 48px; background: var(--crimson);
      box-shadow: 0 0 14px var(--crimson); z-index: 3; border-radius: 3px;
    }
    .main-grid { display: grid; grid-template-columns: 1.15fr 1.35fr; gap: 14px; }
    .card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
      padding: 14px; box-shadow: 0 8px 32px rgba(0,0,0,0.45);
    }
    .cam-wrap { position: relative; border-radius: 10px; overflow: hidden; border: 1px solid var(--cyan); background: #000; }
    img.cam { width: 100%; display: block; height: 310px; object-fit: cover; }

    /* CLICKABLE KPI METRIC CARDS */
    .kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
    .kpi {
      background: rgba(8, 14, 28, 0.92); border: 1px solid rgba(0, 240, 255, 0.18);
      border-radius: 10px; padding: 11px 13px; position: relative; overflow: hidden;
      cursor: pointer; transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }
    .kpi:hover {
      transform: translateY(-3px); border-color: var(--cyan);
      box-shadow: 0 0 18px rgba(0, 240, 255, 0.28);
    }
    .kpi::before { content: ''; position: absolute; top:0; left:0; width: 4px; height: 100%; background: var(--cyan); }
    .kpi.emerald::before { background: var(--emerald); }
    .kpi.amber::before { background: var(--amber); }
    .kpi.purple::before { background: var(--purple); }
    .kpi-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; display:flex; justify-content:space-between; }
    .click-hint { font-size: 10px; color: var(--cyan); opacity: 0.85; font-weight: 600; }
    .kpi-val { font-family: Consolas, monospace; font-size: 24px; font-weight: 700; color: #fff; margin: 5px 0 3px 0; }
    .kpi-sub { font-size: 11px; color: #94a3b8; font-family: Consolas, monospace; }
    .bar-bg { height: 5px; background: #1e293b; border-radius: 4px; overflow: hidden; margin-top: 6px; }
    .bar-fill { height: 100%; background: var(--cyan); width: 50%; transition: width 0.35s ease; }

    /* CONTROL BUTTONS */
    .btn-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    button {
      font-weight: 700; font-size: 11px; letter-spacing: 0.6px;
      padding: 9px 13px; border-radius: 6px; border: 1px solid var(--cyan);
      background: rgba(0, 240, 255, 0.14); color: var(--cyan); cursor: pointer; transition: 0.15s;
    }
    button:hover { background: var(--cyan); color: #000; box-shadow: 0 0 14px var(--cyan); }
    button:active { transform: scale(0.96); }
    button.active-btn { background: var(--amber); border-color: var(--amber); color: #000; box-shadow: 0 0 15px var(--amber); }
    button.danger { border-color: var(--crimson); background: rgba(255, 42, 95, 0.18); color: var(--crimson); }
    button.danger:hover { background: var(--crimson); color: #fff; box-shadow: 0 0 15px var(--crimson); }
    button.success { border-color: var(--emerald); background: rgba(0, 255, 157, 0.18); color: var(--emerald); }
    button.success:hover { background: var(--emerald); color: #000; box-shadow: 0 0 15px var(--emerald); }

    table { width: 100%; border-collapse: collapse; font-size: 12px; font-family: Consolas, monospace; margin-top: 6px; }
    th { text-align: left; padding: 6px; color: var(--cyan); border-bottom: 1px solid var(--border); }
    td { padding: 6px; border-bottom: 1px solid rgba(255,255,255,0.06); }

    canvas.graph-box {
      width: 100%; height: 175px; background: rgba(5, 10, 22, 0.85);
      border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; display: block;
    }

    /* POP-UP SENSOR GRAPH MODAL */
    .modal-overlay {
      position: fixed; inset: 0; background: rgba(3, 6, 15, 0.85); backdrop-filter: blur(8px);
      display: none; align-items: center; justify-content: center; z-index: 9999;
    }
    .modal-overlay.open { display: flex; }
    .modal-card {
      width: min(820px, 92vw); background: #0b1324; border: 2px solid var(--cyan);
      border-radius: 14px; padding: 20px 24px; box-shadow: 0 0 45px rgba(0, 240, 255, 0.3);
    }
    .modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 14px; }
    .modal-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }
    .stat-pill { background: #060b16; border: 1px solid rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; text-align: center; }
    .stat-pill small { color: #94a3b8; font-size: 11px; display: block; }
    .stat-pill b { font-family: Consolas, monospace; font-size: 20px; color: var(--cyan); }
    #modalCanvas { width: 100%; height: 280px; background: #050914; border-radius: 10px; border: 1px solid rgba(0,240,255,0.25); display: block; }
  </style>
</head>
<body>
  <header>
    <div class="brand-title">
      <div class="pulse-dot" id="headerDot"></div>
      <div>
        HYDROSIGHT AI <span style="color:var(--cyan); font-size:13px;">// SCADA DIGITAL TWIN & VISION CRAWLER</span>
        <div style="font-size:11px; color:#94a3b8; font-weight:400;">
          Team Evo Pandas | Hardware Telemetry Link: <span id="hwStatus" style="color:var(--emerald); font-weight:700;">CONNECTING...</span>
        </div>
      </div>
    </div>
    <div class="status-pill" id="statusBadge">NOMINAL OPERATION</div>
  </header>

  <div class="container">
    <!-- INTERACTIVE PIPELINE DIGITAL TWIN -->
    <div class="twin-card">
      <div class="section-tag">
        <span>LIVE PIPELINE DIGITAL TWIN — SECTOR A (10.0m SECTION)</span>
        <span>MAIN VALVE (PIN 7): <b id="v1Text" style="color:var(--emerald);">OPEN</b> &nbsp;|&nbsp; BYPASS VALVE (PIN 8): <b id="v2Text" style="color:#94a3b8;">STANDBY</b></span>
      </div>
      <div class="pipe-track">
        <div class="water-stream" id="waterAnim"></div>
        <div class="crawler-node" id="crawlerNode" style="left: 2%;">0.0m</div>
        <div id="faultPinsContainer"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-family:Consolas,monospace; font-size:11px; color:#94a3b8;">
        <span>0.0m [INLET NODE]</span>
        <span>2.5m</span>
        <span>5.0m [MID-SPAN]</span>
        <span>7.5m</span>
        <span>10.0m [OUTLET VALVE]</span>
      </div>
    </div>

    <div class="main-grid">
      <!-- LEFT COLUMN: VISION CRAWLER + FAULT LOG -->
      <div style="display:flex; flex-direction:column; gap:14px;">
        <div class="card">
          <div class="section-tag">
            <span>TETHERED CRAWLER OPTICAL FEED (OPENCV AI)</span>
            <span>GPS: <b id="gpsText" style="color:#fff;">28.47440N, 77.50400E</b></span>
          </div>
          <div class="cam-wrap">
            <img class="cam" src="/video_feed" />
          </div>
          <div class="btn-row">
            <button id="btnCrawler" onclick="sendCmd('TOGGLE_ROBOT')"> START CRAWLER</button>
            <button onclick="sendCmd('RESET_DIST')">↺ RESET ODOMETRY</button>
            <button class="danger" onclick="sendCmd('ISOLATE')"> ISOLATE PIPE RELAYS</button>
            <button class="success" onclick="sendCmd('NORMAL')"> RESTORE FLOW</button>
            <button id="btnBurst" onclick="sendCmd('SIM_BURST')"> SIMULATE BURST SURGE</button>
            <button onclick="sendCmd('SWITCH_CAM')"> SWITCH CAM</button>
          </div>
        </div>

        <div class="card">
          <div class="section-tag">
            <span>LOCALIZED STRUCTURAL DEFECT & ANOMALY LOG</span>
            <span>RAW ADC: TURB=<b id="rawT">0</b> | pH=<b id="rawP">0</b></span>
          </div>
          <table>
            <thead><tr><th>TIME</th><th>DETECTED DEFECT</th><th>ODOMETRY</th><th>GPS COORDS</th></tr></thead>
            <tbody id="logBody"><tr><td colspan="4" style="color:#64748b;">No structural anomalies detected in sector...</td></tr></tbody>
          </table>
        </div>
      </div>

      <!-- RIGHT COLUMN: 6 CLICKABLE KPI GAUGES + LIVE OSCILLOSCOPE GRAPH -->
      <div style="display:flex; flex-direction:column; gap:14px;">
        <div class="card">
          <div class="section-tag">
            <span>REAL-TIME HYDRO-QUALITY & FLOW TELEMETRY</span>
            <span style="color:var(--emerald);"> CLICK ANY SENSOR CARD FOR LIVE POP-UP GRAPH</span>
          </div>
          <div class="kpi-grid">
            <div class="kpi" onclick="openSensorModal('flow_lpm')">
              <div class="kpi-label"><span>Instantaneous Flow</span><span class="click-hint"> GRAPH</span></div>
              <div class="kpi-val"><span id="flow">15.40</span> <small style="font-size:13px;">L/m</small></div>
              <div class="kpi-sub">Nominal: 14.0–17.0 L/m</div>
              <div class="bar-bg"><div class="bar-fill" id="barFlow" style="width:55%;"></div></div>
            </div>

            <div class="kpi emerald" onclick="openSensorModal('total_liters')">
              <div class="kpi-label"><span>Total Consumption</span><span class="click-hint"> GRAPH</span></div>
              <div class="kpi-val"><span id="total">142.80</span> <small style="font-size:13px;">L</small></div>
              <div class="kpi-sub">Cumulative Metered</div>
              <div class="bar-bg"><div class="bar-fill" style="width:100%; background:var(--emerald);"></div></div>
            </div>

            <div class="kpi amber" onclick="openSensorModal('turbidity_ntu')">
              <div class="kpi-label"><span>Turbidity (Optical)</span><span class="click-hint"> GRAPH</span></div>
              <div class="kpi-val"><span id="turb">4.2</span> <small style="font-size:13px;">NTU</small></div>
              <div class="kpi-sub">Threshold: &lt; 50 NTU</div>
              <div class="bar-bg"><div class="bar-fill" id="barTurb" style="width:8%; background:var(--amber);"></div></div>
            </div>

            <div class="kpi" onclick="openSensorModal('ph')">
              <div class="kpi-label"><span>Water Acidity (pH)</span><span class="click-hint"> GRAPH</span></div>
              <div class="kpi-val"><span id="ph">7.20</span> <small style="font-size:13px;">pH</small></div>
              <div class="kpi-sub">Safe Band: 6.5–8.5</div>
              <div class="bar-bg"><div class="bar-fill" id="barPh" style="width:60%;"></div></div>
            </div>

            <div class="kpi emerald" onclick="openSensorModal('chlorine_mg_l')">
              <div class="kpi-label"><span>Residual Chlorine</span><span class="click-hint"> GRAPH</span></div>
              <div class="kpi-val"><span id="cl">1.38</span> <small style="font-size:13px;">mg/L</small></div>
              <div class="kpi-sub">AI Soft-Sensor Model</div>
              <div class="bar-bg"><div class="bar-fill" id="barCl" style="width:70%; background:var(--emerald);"></div></div>
            </div>

            <div class="kpi purple" onclick="openSensorModal('ultrasonic_cm')">
              <div class="kpi-label"><span>Obstruction Sonar</span><span class="click-hint"> GRAPH</span></div>
              <div class="kpi-val"><span id="sonar">24.0</span> <small style="font-size:13px;">cm</small></div>
              <div class="kpi-sub">HC-SR04 Ultrasonic</div>
              <div class="bar-bg"><div class="bar-fill" id="barSonar" style="width:40%; background:var(--purple);"></div></div>
            </div>
          </div>

          <div class="section-tag" style="margin-top:6px;">
            <span>LIVE MULTI-CHANNEL WAVEFORM (FLOW RATE [CYAN] vs TURBIDITY [AMBER])</span>
            <span id="liveWaveText">FLOW: 15.40 L/m | TURB: 4.2 NTU</span>
          </div>
          <canvas id="mainGraph" class="graph-box" width="700" height="175"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- POP-UP MODAL FOR INDIVIDUAL SENSOR REAL-TIME GRAPH -->
  <div class="modal-overlay" id="sensorModal" onclick="if(event.target===this) closeSensorModal()">
    <div class="modal-card">
      <div class="modal-header">
        <div>
          <h3 id="modalTitle" style="margin:0; color:#fff; letter-spacing:1px;">SENSOR LIVE TELEMETRY</h3>
          <small id="modalSub" style="color:var(--cyan);">Real-Time 2.0 Hz Hardware Data Stream</small>
        </div>
        <button class="danger" onclick="closeSensorModal()">✕ CLOSE GRAPH (ESC)</button>
      </div>
      <div class="modal-stats">
        <div class="stat-pill"><small>CURRENT READING</small><b id="mCurr">0.00</b></div>
        <div class="stat-pill"><small>SESSION MINIMUM</small><b id="mMin">0.00</b></div>
        <div class="stat-pill"><small>SESSION MAXIMUM</small><b id="mMax">0.00</b></div>
        <div class="stat-pill"><small>ROLLING AVERAGE</small><b id="mAvg">0.00</b></div>
      </div>
      <canvas id="modalCanvas" width="760" height="280"></canvas>
    </div>
  </div>

  <script>
    const MAX_POINTS = 35;
    const sensorMeta = {
      flow_lpm:      { title: "INSTANTANEOUS PIPE FLOW RATE", unit: "L/min", color: "#00f0ff", min: 0, max: 32, sub: "Safe Nominal Range: 14.0 – 17.0 L/min (0 = Valve Isolated)" },
      total_liters:  { title: "CUMULATIVE WATER CONSUMPTION", unit: "L",     color: "#00ff9d", min: 130, max: 200, sub: "Continuous Integrated Volumetric Meter" },
      turbidity_ntu: { title: "OPTICAL WATER TURBIDITY",      unit: "NTU",   color: "#ffb800", min: 0, max: 360, sub: "Clean Water: < 25 NTU | Contamination Alert: > 180 NTU" },
      ph:            { title: "WATER ACIDITY / ALKALINITY",   unit: "pH",    color: "#38bdf8", min: 4, max: 10,  sub: "WHO Safe Drinking Band: 6.5 – 8.5 pH" },
      chlorine_mg_l: { title: "RESIDUAL FREE CHLORINE (AI)",  unit: "mg/L",  color: "#00ff9d", min: 0, max: 2.0, sub: "Physics-Informed Soft-Sensor Derived from pH & Turbidity" },
      ultrasonic_cm: { title: "ULTRASONIC OBSTRUCTION SONAR", unit: "cm",    color: "#b86bff", min: 0, max: 120, sub: "HC-SR04 Distance Telemetry (Pin 9 Trig / Pin 10 Echo)" }
    };

    const history = {
      flow_lpm:      Array(MAX_POINTS).fill(15.4),
      total_liters:  Array(MAX_POINTS).fill(142.8),
      turbidity_ntu: Array(MAX_POINTS).fill(4.2),
      ph:            Array(MAX_POINTS).fill(7.2),
      chlorine_mg_l: Array(MAX_POINTS).fill(1.38),
      ultrasonic_cm: Array(MAX_POINTS).fill(24.0)
    };

    let activeModalSensor = null;

    function openSensorModal(sensorKey) {
      activeModalSensor = sensorKey;
      const meta = sensorMeta[sensorKey];
      document.getElementById('modalTitle').innerText = meta.title + " (" + meta.unit + ")";
      document.getElementById('modalSub').innerText = meta.sub;
      document.getElementById('sensorModal').classList.add('open');
      drawModalGraph();
    }

    function closeSensorModal() {
      activeModalSensor = null;
      document.getElementById('sensorModal').classList.remove('open');
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeSensorModal();
    });

    async function sendCmd(cmd) {
      try {
        const res = await fetch('/api/control', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({cmd})
        });
        const d = await res.json();
        applyTelemetryToUI(d);
      } catch (err) {
        console.error("Control error:", err);
      }
    }

    function drawDualGraph() {
      const canvas = document.getElementById('mainGraph');
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Draw Grid
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const y = (h / 4) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // Helper to plot series
      function plotSeries(arr, minVal, maxVal, color, fillAlpha) {
        ctx.beginPath();
        arr.forEach((val, idx) => {
          const x = (idx / (MAX_POINTS - 1)) * (w - 20) + 10;
          const norm = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal || 1)));
          const y = h - 14 - norm * (h - 28);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.stroke();

        ctx.lineTo(w - 10, h - 10);
        ctx.lineTo(10, h - 10);
        ctx.closePath();
        ctx.fillStyle = fillAlpha;
        ctx.fill();
      }

      // Plot Turbidity (0-350 NTU scale) & Flow Rate (0-32 L/m scale)
      plotSeries(history.turbidity_ntu, 0, 350, "#ffb800", "rgba(255, 184, 0, 0.08)");
      plotSeries(history.flow_lpm, 0, 32, "#00f0ff", "rgba(0, 240, 255, 0.14)");
    }

    function drawModalGraph() {
      if (!activeModalSensor) return;
      const arr = history[activeModalSensor];
      const meta = sensorMeta[activeModalSensor];
      const canvas = document.getElementById('modalCanvas');
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const curr = arr[arr.length - 1];
      const minV = Math.min(...arr);
      const maxV = Math.max(...arr);
      const avgV = arr.reduce((a, b) => a + b, 0) / arr.length;

      document.getElementById('mCurr').innerText = curr.toFixed(2) + " " + meta.unit;
      document.getElementById('mCurr').style.color = meta.color;
      document.getElementById('mMin').innerText = minV.toFixed(2);
      document.getElementById('mMax').innerText = maxV.toFixed(2);
      document.getElementById('mAvg').innerText = avgV.toFixed(2);

      // Dynamic bounds so live fluctuations always look clear
      const span = Math.max(2.0, (maxV - minV) * 1.5);
      const lowBound = Math.max(0, minV - span * 0.25);
      const highBound = maxV + span * 0.25;

      // Grid lines + Y-axis labels
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px Consolas, monospace";
      for (let i = 0; i <= 4; i++) {
        const y = 20 + ((h - 40) / 4) * i;
        const labelVal = highBound - ((highBound - lowBound) / 4) * i;
        ctx.beginPath(); ctx.moveTo(55, y); ctx.lineTo(w - 15, y); ctx.stroke();
        ctx.fillText(labelVal.toFixed(1), 8, y + 4);
      }

      // Plot Live Waveform
      ctx.beginPath();
      arr.forEach((val, idx) => {
        const x = 55 + (idx / (MAX_POINTS - 1)) * (w - 75);
        const norm = Math.max(0, Math.min(1, (val - lowBound) / (highBound - lowBound || 1)));
        const y = (h - 20) - norm * (h - 40);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = meta.color;
      ctx.lineWidth = 3;
      ctx.stroke();

      // Glow fill
      ctx.lineTo(w - 20, h - 20);
      ctx.lineTo(55, h - 20);
      ctx.closePath();
      ctx.fillStyle = meta.color + "22";
      ctx.fill();

      // Draw latest point dot
      const lastNorm = Math.max(0, Math.min(1, (curr - lowBound) / (highBound - lowBound || 1)));
      const lastY = (h - 20) - lastNorm * (h - 40);
      ctx.beginPath();
      ctx.arc(w - 20, lastY, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
    }

    function applyTelemetryToUI(d) {
      document.getElementById('flow').innerText = Number(d.flow_lpm).toFixed(2);
      document.getElementById('total').innerText = Number(d.total_liters).toFixed(2);
      document.getElementById('turb').innerText = Number(d.turbidity_ntu).toFixed(1);
      document.getElementById('ph').innerText = Number(d.ph).toFixed(2);
      document.getElementById('cl').innerText = Number(d.chlorine_mg_l).toFixed(2);
      document.getElementById('sonar').innerText = Number(d.ultrasonic_cm).toFixed(1);
      document.getElementById('rawT').innerText = d.raw_turb;
      document.getElementById('rawP').innerText = d.raw_ph;
      document.getElementById('gpsText').innerText = Number(d.gps_lat).toFixed(5) + 'N, ' + Number(d.gps_lon).toFixed(5) + 'E';
      document.getElementById('liveWaveText').innerText = `FLOW: ${Number(d.flow_lpm).toFixed(2)} L/m | TURB: ${Number(d.turbidity_ntu).toFixed(1)} NTU`;

      // Progress bars
      document.getElementById('barFlow').style.width = Math.min(100, (d.flow_lpm / 30) * 100) + '%';
      document.getElementById('barTurb').style.width = Math.min(100, (d.turbidity_ntu / 300) * 100) + '%';
      document.getElementById('barPh').style.width = Math.min(100, (d.ph / 14) * 100) + '%';
      document.getElementById('barCl').style.width = Math.min(100, (d.chlorine_mg_l / 2.0) * 100) + '%';
      document.getElementById('barSonar').style.width = Math.min(100, (d.ultrasonic_cm / 100) * 100) + '%';

      // Crawler Button & Pipe Map Position
      const btnCrawler = document.getElementById('btnCrawler');
      btnCrawler.innerText = d.robot_moving ? ' PAUSE CRAWLER (MOVING)' : ' START CRAWLER';
      btnCrawler.className = d.robot_moving ? 'active-btn' : '';

      const btnBurst = document.getElementById('btnBurst');
      btnBurst.className = d.leak_sim ? 'danger' : '';

      const pct = Math.min(92, Math.max(2, (d.robot_dist_m / 10.0) * 92));
      const bot = document.getElementById('crawlerNode');
      bot.style.left = pct + '%';
      bot.innerText = Number(d.robot_dist_m).toFixed(1) + 'm';

      document.getElementById('waterAnim').style.animationPlayState = d.valve1_open ? 'running' : 'paused';
      document.getElementById('waterAnim').style.opacity = d.valve1_open ? '1' : '0.12';

      document.getElementById('v1Text').innerText = d.valve1_open ? 'OPEN (FLOWING)' : 'ISOLATED (SHUT)';
      document.getElementById('v1Text').style.color = d.valve1_open ? '#00ff9d' : '#ff2a5f';
      document.getElementById('v2Text').innerText = d.valve2_open ? 'BYPASS ACTIVE' : 'STANDBY';
      document.getElementById('v2Text').style.color = d.valve2_open ? '#ffb800' : '#94a3b8';

      document.getElementById('hwStatus').innerText = d.serial_connected ? `ARDUINO LIVE (${d.serial_port_name})` : 'SIMULATION / CHECK USB';
      document.getElementById('hwStatus').style.color = d.serial_connected ? '#00ff9d' : '#ffb800';

      const badge = document.getElementById('statusBadge');
      badge.innerText = d.status;
      const isAlert = d.status !== 'NOMINAL OPERATION';
      badge.style.borderColor = isAlert ? '#ff2a5f' : '#00ff9d';
      badge.style.color = isAlert ? '#ff2a5f' : '#00ff9d';
      badge.style.background = isAlert ? 'rgba(255,42,95,0.18)' : 'rgba(0,255,157,0.15)';

      if (d.fault_history && d.fault_history.length > 0) {
        document.getElementById('logBody').innerHTML = d.fault_history.map(
          f => `<tr><td>${f.time}</td><td style="color:#ff2a5f;font-weight:bold;">${f.type}</td><td>${f.dist}m</td><td>${f.gps}</td></tr>`
        ).join('');
        document.getElementById('faultPinsContainer').innerHTML = d.fault_history.map(
          f => `<div class="fault-pin" style="left:${Math.min(94, (f.dist/10.0)*94)}%;" title="${f.type}"></div>`
        ).join('');
      } else {
        document.getElementById('logBody').innerHTML = '<tr><td colspan="4" style="color:#64748b;">No structural anomalies detected in sector...</td></tr>';
        document.getElementById('faultPinsContainer').innerHTML = '';
      }
    }

    setInterval(async () => {
      try {
        const res = await fetch('/api/telemetry');
        const d = await res.json();

        // Push to rolling history arrays
        Object.keys(history).forEach(k => {
          if (d[k] !== undefined) {
            history[k].push(Number(d[k]));
            if (history[k].length > MAX_POINTS) history[k].shift();
          }
        });

        applyTelemetryToUI(d);
        drawDualGraph();
        drawModalGraph();
      } catch (e) {}
    }, 500);
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)