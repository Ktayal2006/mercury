#!/usr/bin/env bash
# =============================================================================
# setup_face_recognition.sh
#
# Run this script from the ROOT of your ugv-dtu-mercury repo:
#   chmod +x setup_face_recognition.sh
#   ./setup_face_recognition.sh
#
# What it does:
#   1. Creates git branch: feature/face-recognition-plan3
#   2. Creates face_recognition_node.py  (vision: detect + recognize)
#   3. Creates face_task_node.py          (brain: state machine, 18-image grid)
#   4. Creates turret_controller_node.py  (hardware bridge: servos + laser)
#   5. Creates face_task.launch.py        (launch file)
#   6. Creates arduino/turret_bridge.ino  (Arduino firmware)
#   7. Patches src/perception/setup.py   (adds 3 console scripts)
#   8. Patches src/perception/package.xml (adds cv_bridge depend)
#   9. Patches Dockerfile                 (adds pip installs)
# =============================================================================

set -e  # exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Sanity check ──────────────────────────────────────────────────────────────
[ -f "docker-compose.yml" ] || err "Run this script from the root of ugv-dtu-mercury repo"
[ -d "src/perception" ]     || err "src/perception directory not found"

# ── 1. Git branch ─────────────────────────────────────────────────────────────
log "Creating git branch: feature/face-recognition-plan3"
git checkout -b feature/face-recognition-plan3 2>/dev/null || {
    warn "Branch already exists, switching to it"
    git checkout feature/face-recognition-plan3
}

# ── 2. Create directories ─────────────────────────────────────────────────────
mkdir -p arduino/turret_bridge
mkdir -p src/perception/launch
log "Directories ready"

# =============================================================================
# FILE 1: face_recognition_node.py
# Role: Pure vision node. On request, grabs one camera frame, runs YOLOv8-face
#       detection + ArcFace recognition, returns result.
# =============================================================================
log "Writing face_recognition_node.py"
cat > src/perception/perception/face_recognition_node.py << 'PYEOF'
#!/usr/bin/env python3
"""
face_recognition_node.py
========================
Pure vision node for Plan 3 (18-image grid scan).

Subscribes:
    /camera/image_raw          (sensor_msgs/Image)
    /face/capture_request      (std_msgs/Bool)  -- True triggers one capture+match

Publishes:
    /face/match_found          (std_msgs/Bool)
    /face/horizontal_error     (std_msgs/Float32)  pixels from image centre (+ = right)
    /face/vertical_error       (std_msgs/Float32)  pixels from image centre (+ = down)
    /face/best_similarity      (std_msgs/Float32)  for debug/tuning

Parameters:
    target_image_path  (string)  path to pre-given target face image
    similarity_threshold (float) default 0.35
    detection_size       (int)   InsightFace input size, default 320
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class FaceRecognitionNode(Node):

    def __init__(self):
        super().__init__('face_recognition_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('target_image_path', '')
        self.declare_parameter('similarity_threshold', 0.35)
        self.declare_parameter('detection_size', 320)

        self.target_path  = self.get_parameter('target_image_path').value
        self.threshold    = self.get_parameter('similarity_threshold').value
        det_size          = self.get_parameter('detection_size').value

        # ── InsightFace model ─────────────────────────────────────────────────
        self.app          = None
        self.target_emb   = None
        self._det_size    = (det_size, det_size)
        self._load_model()

        # ── State ─────────────────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self._latest_frame = None
        self._capture_requested = False

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Image, '/camera/image_raw',
                                 self._image_cb, 10)
        self.create_subscription(Bool,  '/face/capture_request',
                                 self._capture_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_match  = self.create_publisher(Bool,    '/face/match_found',      10)
        self._pub_herr   = self.create_publisher(Float32, '/face/horizontal_error', 10)
        self._pub_verr   = self.create_publisher(Float32, '/face/vertical_error',   10)
        self._pub_sim    = self.create_publisher(Float32, '/face/best_similarity',  10)

        self.get_logger().info('FaceRecognitionNode ready.')

    # ──────────────────────────────────────────────────────────────────────────
    def _load_model(self):
        """Load InsightFace model and encode target face embedding."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            self.get_logger().error(
                'insightface not installed! Run: pip install insightface onnxruntime-gpu')
            return

        self.get_logger().info('Loading InsightFace buffalo_sc model...')
        self.app = FaceAnalysis(
            name='buffalo_sc',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=self._det_size)

        # GPU warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.app.get(dummy)
        self.get_logger().info('Model loaded and warmed up.')

        # Encode target image
        if self.target_path:
            self._encode_target()
        else:
            self.get_logger().warn(
                'No target_image_path set. Publish to /face/capture_request '
                'only after setting the parameter.')

    def _encode_target(self):
        """Encode target face image into a 512-D ArcFace embedding."""
        img = cv2.imread(self.target_path)
        if img is None:
            self.get_logger().error(f'Cannot read target image: {self.target_path}')
            return
        faces = self.app.get(img)
        if not faces:
            self.get_logger().error('No face found in target image!')
            return
        # Use largest face
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        self.target_emb = largest.normed_embedding
        self.get_logger().info(
            f'Target face encoded. Embedding shape: {self.target_emb.shape}')

    # ──────────────────────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        """Cache the latest camera frame."""
        try:
            self._latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')

    def _capture_cb(self, msg: Bool):
        """On True signal, process the latest frame immediately."""
        if msg.data:
            self._process_frame()

    # ──────────────────────────────────────────────────────────────────────────
    def _process_frame(self):
        """
        Run face detection + ArcFace recognition on the latest cached frame.
        Publishes match result and pixel errors.
        """
        if self.app is None:
            self.get_logger().warn('Model not loaded yet.')
            return
        if self.target_emb is None:
            self.get_logger().warn('Target embedding not loaded yet.')
            return
        if self._latest_frame is None:
            self.get_logger().warn('No camera frame received yet.')
            return

        frame = self._latest_frame.copy()
        h, w  = frame.shape[:2]

        t0    = time.time()
        faces = self.app.get(frame)
        dt    = (time.time() - t0) * 1000

        if not faces:
            self.get_logger().info(f'[{dt:.0f}ms] No faces detected in frame.')
            self._pub_match.publish(Bool(data=False))
            self._pub_sim.publish(Float32(data=0.0))
            return

        best_sim  = -1.0
        best_face = None

        for face in faces:
            sim = float(np.dot(self.target_emb, face.normed_embedding))
            if sim > best_sim:
                best_sim  = sim
                best_face = face

        matched = best_sim >= self.threshold

        # Pixel errors from image centre
        cx = (best_face.bbox[0] + best_face.bbox[2]) / 2.0
        cy = (best_face.bbox[1] + best_face.bbox[3]) / 2.0
        h_err = cx - (w / 2.0)   # + = face is right of centre
        v_err = cy - (h / 2.0)   # + = face is below centre

        self.get_logger().info(
            f'[{dt:.0f}ms] {len(faces)} face(s) | best_sim={best_sim:.3f} '
            f'| matched={matched} | h_err={h_err:.1f}px v_err={v_err:.1f}px')

        self._pub_match.publish(Bool(data=matched))
        self._pub_herr.publish(Float32(data=float(h_err)))
        self._pub_verr.publish(Float32(data=float(v_err)))
        self._pub_sim.publish(Float32(data=float(best_sim)))


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
PYEOF
log "face_recognition_node.py written"


# =============================================================================
# FILE 2: face_task_node.py
# Role: State machine brain. Controls the full Plan 3 sequence:
#       IDLE → SCANNING (18-image grid) → FINE_TUNE → FIRE → DONE
# =============================================================================
log "Writing face_task_node.py"
cat > src/perception/perception/face_task_node.py << 'PYEOF'
#!/usr/bin/env python3
"""
face_task_node.py
=================
Brain node for Plan 3 face recognition task.

State machine:
    IDLE        -- waiting for /face_task/start
    SCANNING    -- iterating through 18 (H×V) turret positions, capturing + recognising
    FINE_TUNE   -- proportional controller to centre face in frame
    FIRE        -- activate laser, wait, publish done
    DONE        -- publish /face_task/complete, return to IDLE

Subscribes:
    /face_task/start           (std_msgs/Bool)   -- True starts the task
    /face/match_found          (std_msgs/Bool)
    /face/horizontal_error     (std_msgs/Float32)
    /face/vertical_error       (std_msgs/Float32)

Publishes:
    /face/capture_request      (std_msgs/Bool)   -- trigger vision node
    /turret/pan_deg            (std_msgs/Float32) -- horizontal servo target (degrees)
    /turret/tilt_deg           (std_msgs/Float32) -- vertical servo target (degrees)
    /laser/fire                (std_msgs/Bool)
    /face_task/complete        (std_msgs/Bool)   -- True when task finished

Parameters (all tunable at launch):
    h_positions_deg   list of 6 horizontal angles  default: [-75,-45,-15,15,45,75]
    v_positions_deg   list of 3 vertical angles    default: [40,25,59]  (mid,low,high)
    settle_time_sec   turret settle delay           default: 0.5
    fine_tune_px_tol  pixel tolerance for fine-tune default: 20.0
    fine_tune_gain_h  proportional gain horizontal  default: 0.05 (deg/px)
    fine_tune_gain_v  proportional gain vertical    default: 0.05 (deg/px)
    fine_tune_timeout timeout for fine-tune phase   default: 5.0
    laser_on_time_sec laser fire duration           default: 3.0
    pan_centre_deg    servo centre for pan          default: 90.0
    tilt_centre_deg   servo centre for tilt         default: 90.0
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
import time


# ── State constants ────────────────────────────────────────────────────────────
IDLE       = 'IDLE'
SCANNING   = 'SCANNING'
FINE_TUNE  = 'FINE_TUNE'
FIRE       = 'FIRE'
DONE       = 'DONE'


class FaceTaskNode(Node):

    def __init__(self):
        super().__init__('face_task_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('h_positions_deg',   [-75.0, -45.0, -15.0, 15.0, 45.0, 75.0])
        self.declare_parameter('v_positions_deg',   [40.0, 25.0, 59.0])   # mid, low, high
        self.declare_parameter('settle_time_sec',   0.5)
        self.declare_parameter('fine_tune_px_tol',  20.0)
        self.declare_parameter('fine_tune_gain_h',  0.05)
        self.declare_parameter('fine_tune_gain_v',  0.05)
        self.declare_parameter('fine_tune_timeout', 5.0)
        self.declare_parameter('laser_on_time_sec', 3.0)
        self.declare_parameter('pan_centre_deg',    90.0)
        self.declare_parameter('tilt_centre_deg',   90.0)

        self._h_pos      = self.get_parameter('h_positions_deg').value
        self._v_pos      = self.get_parameter('v_positions_deg').value
        self._settle     = self.get_parameter('settle_time_sec').value
        self._tol        = self.get_parameter('fine_tune_px_tol').value
        self._gain_h     = self.get_parameter('fine_tune_gain_h').value
        self._gain_v     = self.get_parameter('fine_tune_gain_v').value
        self._ft_timeout = self.get_parameter('fine_tune_timeout').value
        self._laser_time = self.get_parameter('laser_on_time_sec').value
        self._pan_ctr    = self.get_parameter('pan_centre_deg').value
        self._tilt_ctr   = self.get_parameter('tilt_centre_deg').value

        # ── Build 18-position scan grid ────────────────────────────────────────
        # Order: V[0](mid) full H sweep → V[1](low) full H sweep → V[2](high) full H sweep
        self._grid = []
        for v in self._v_pos:
            for h in self._h_pos:
                self._grid.append((h, v))
        self.get_logger().info(
            f'Scan grid: {len(self._grid)} positions | '
            f'H={self._h_pos} | V={self._v_pos}')

        # ── State ─────────────────────────────────────────────────────────────
        self._state          = IDLE
        self._grid_idx       = 0
        self._waiting_result = False
        self._match_found    = False
        self._h_err          = 0.0
        self._v_err          = 0.0
        self._ft_start       = None
        self._fire_start     = None

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Bool,    '/face_task/start',        self._start_cb, 10)
        self.create_subscription(Bool,    '/face/match_found',       self._match_cb, 10)
        self.create_subscription(Float32, '/face/horizontal_error',  self._herr_cb,  10)
        self.create_subscription(Float32, '/face/vertical_error',    self._verr_cb,  10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_capture  = self.create_publisher(Bool,    '/face/capture_request', 10)
        self._pub_pan      = self.create_publisher(Float32, '/turret/pan_deg',       10)
        self._pub_tilt     = self.create_publisher(Float32, '/turret/tilt_deg',      10)
        self._pub_laser    = self.create_publisher(Bool,    '/laser/fire',           10)
        self._pub_complete = self.create_publisher(Bool,    '/face_task/complete',   10)

        # ── Main control loop at 10 Hz ─────────────────────────────────────────
        self._loop_timer = self.create_timer(0.1, self._loop)

        self.get_logger().info('FaceTaskNode ready. Waiting for /face_task/start ...')

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _start_cb(self, msg: Bool):
        if msg.data and self._state == IDLE:
            self.get_logger().info('Task START received. Beginning 18-image grid scan.')
            self._reset_scan()
            self._state = SCANNING

    def _match_cb(self, msg: Bool):
        if self._waiting_result:
            self._match_found    = msg.data
            self._waiting_result = False

    def _herr_cb(self, msg: Float32):
        self._h_err = msg.data

    def _verr_cb(self, msg: Float32):
        self._v_err = msg.data

    # ── Main loop ─────────────────────────────────────────────────────────────
    def _loop(self):
        if self._state == IDLE:
            return

        elif self._state == SCANNING:
            self._run_scanning()

        elif self._state == FINE_TUNE:
            self._run_fine_tune()

        elif self._state == FIRE:
            self._run_fire()

        elif self._state == DONE:
            pass  # stay here until reset

    # ── SCANNING phase ────────────────────────────────────────────────────────
    def _run_scanning(self):
        """
        Step through the 18-position grid one position at a time.
        For each position:
          1. Command turret to (H, V)
          2. Wait settle_time
          3. Trigger capture
          4. Wait for match result
          5. If match → transition to FINE_TUNE
          6. Else → advance to next position
        """
        # If we're waiting for vision node to respond, do nothing
        if self._waiting_result:
            return

        # All positions exhausted without finding target
        if self._grid_idx >= len(self._grid):
            self.get_logger().warn(
                'Scan complete — target NOT found in all 18 positions. '
                'Publishing task complete (no match).')
            self._pub_complete.publish(Bool(data=False))
            self._state = DONE
            return

        h_deg, v_deg = self._grid[self._grid_idx]

        # ── Substep tracking via a small internal sub-state ────────────────
        # We use _scan_substep: 0=move, 1=settling, 2=capture, 3=wait_result
        if not hasattr(self, '_scan_substep'):
            self._scan_substep  = 0
            self._scan_step_t   = None

        if self._scan_substep == 0:
            # Command turret
            self._move_turret(h_deg, v_deg)
            self._scan_substep = 1
            self._scan_step_t  = time.time()
            self.get_logger().info(
                f'[{self._grid_idx+1:02d}/18] Moving turret → H={h_deg}° V={v_deg}°')

        elif self._scan_substep == 1:
            # Wait for settle
            if time.time() - self._scan_step_t >= self._settle:
                self._scan_substep = 2

        elif self._scan_substep == 2:
            # Trigger capture + recognition
            self._waiting_result = True
            self._match_found    = False
            self._pub_capture.publish(Bool(data=True))
            self._scan_substep = 3

        elif self._scan_substep == 3:
            # Result arrived (_waiting_result cleared by _match_cb)
            if self._match_found:
                self.get_logger().info(
                    f'TARGET FOUND at grid position [{self._grid_idx+1}/18] '
                    f'H={h_deg}° V={v_deg}°. Transitioning to FINE_TUNE.')
                self._state        = FINE_TUNE
                self._ft_start     = time.time()
                self._scan_substep = 0
            else:
                # Advance to next position
                self._grid_idx    += 1
                self._scan_substep = 0

    # ── FINE_TUNE phase ───────────────────────────────────────────────────────
    def _run_fine_tune(self):
        """
        Proportional controller: adjust turret until face is centred
        within ±_tol pixels on both axes.
        Timeout after _ft_timeout seconds → fire anyway.
        """
        elapsed = time.time() - self._ft_start

        # Trigger a fresh capture every 0.2s to get updated pixel errors
        if not hasattr(self, '_ft_last_capture'):
            self._ft_last_capture = 0.0

        if time.time() - self._ft_last_capture >= 0.2:
            self._pub_capture.publish(Bool(data=True))
            self._ft_last_capture = time.time()

        h_ok = abs(self._h_err) <= self._tol
        v_ok = abs(self._v_err) <= self._tol

        if (h_ok and v_ok) or elapsed >= self._ft_timeout:
            reason = 'centred' if (h_ok and v_ok) else 'timeout'
            self.get_logger().info(
                f'Fine-tune complete ({reason}). '
                f'h_err={self._h_err:.1f}px v_err={self._v_err:.1f}px')
            # Clean up
            if hasattr(self, '_ft_last_capture'):
                del self._ft_last_capture
            self._state      = FIRE
            self._fire_start = time.time()
            return

        # Proportional correction
        # h_err > 0 → face is right → increase pan (positive direction)
        # v_err > 0 → face is below → increase tilt
        current_h, current_v = self._grid[self._grid_idx]
        new_h = current_h + self._gain_h * self._h_err
        new_v = current_v + self._gain_v * self._v_err

        # Clamp to safe range
        new_h = max(-90.0, min(90.0, new_h))
        new_v = max(0.0,   min(120.0, new_v))

        self._move_turret(new_h, new_v)

    # ── FIRE phase ────────────────────────────────────────────────────────────
    def _run_fire(self):
        """Fire laser for laser_on_time_sec, then publish complete."""
        if time.time() - self._fire_start < self._laser_time:
            self._pub_laser.publish(Bool(data=True))
        else:
            self._pub_laser.publish(Bool(data=False))
            self.get_logger().info('Laser OFF. Task COMPLETE.')
            self._pub_complete.publish(Bool(data=True))
            self._state = DONE

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _move_turret(self, pan_deg: float, tilt_deg: float):
        """Publish turret position commands."""
        # Convert from scan angles to servo angles
        # pan_centre_deg is the servo neutral (facing forward)
        servo_pan  = self._pan_ctr  + pan_deg   # add offset from centre
        servo_tilt = self._tilt_ctr + tilt_deg  # add offset from centre
        self._pub_pan.publish(Float32(data=float(servo_pan)))
        self._pub_tilt.publish(Float32(data=float(servo_tilt)))

    def _reset_scan(self):
        """Reset state for a fresh scan."""
        self._grid_idx       = 0
        self._waiting_result = False
        self._match_found    = False
        self._h_err          = 0.0
        self._v_err          = 0.0
        self._ft_start       = None
        self._fire_start     = None
        if hasattr(self, '_scan_substep'):
            del self._scan_substep
        if hasattr(self, '_ft_last_capture'):
            del self._ft_last_capture


def main(args=None):
    rclpy.init(args=args)
    node = FaceTaskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
PYEOF
log "face_task_node.py written"


# =============================================================================
# FILE 3: turret_controller_node.py
# Role: Hardware bridge. Converts ROS2 degree commands to PWM via serial
#       to an Arduino/ESP32. Also handles laser GPIO.
# =============================================================================
log "Writing turret_controller_node.py"
cat > src/perception/perception/turret_controller_node.py << 'PYEOF'
#!/usr/bin/env python3
"""
turret_controller_node.py
=========================
Hardware bridge node. Converts ROS2 servo angle commands to PWM signals
sent over serial to an Arduino (or ESP32).

Serial protocol (simple ASCII, one command per line):
    P<angle>    -- set pan servo   e.g. "P95.5\n"
    T<angle>    -- set tilt servo  e.g. "T112.0\n"
    L1          -- laser ON
    L0          -- laser OFF

Subscribes:
    /turret/pan_deg    (std_msgs/Float32)  servo angle in degrees 0-180
    /turret/tilt_deg   (std_msgs/Float32)  servo angle in degrees 0-180
    /laser/fire        (std_msgs/Bool)

Parameters:
    serial_port    default: /dev/ttyUSB1  (change to your Arduino port)
    baud_rate      default: 115200
    dry_run        default: true   (if true: log commands but don't open serial)
                                   Set to false on real hardware
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
import time


class TurretControllerNode(Node):

    def __init__(self):
        super().__init__('turret_controller_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('serial_port', '/dev/ttyUSB1')
        self.declare_parameter('baud_rate',   115200)
        self.declare_parameter('dry_run',     True)

        port     = self.get_parameter('serial_port').value
        baud     = self.get_parameter('baud_rate').value
        self._dry = self.get_parameter('dry_run').value

        # ── Serial connection ─────────────────────────────────────────────────
        self._serial = None
        if not self._dry:
            try:
                import serial
                self._serial = serial.Serial(port, baud, timeout=0.1)
                time.sleep(2.0)  # wait for Arduino reset
                self.get_logger().info(f'Serial connected: {port} @ {baud}')
            except Exception as e:
                self.get_logger().error(f'Serial open failed: {e}')
                self.get_logger().warn('Falling back to dry_run mode.')
                self._dry = True
        else:
            self.get_logger().warn(
                f'dry_run=True — serial commands will be LOGGED ONLY, not sent. '
                f'Set dry_run:=false on real hardware.')

        # ── State ─────────────────────────────────────────────────────────────
        self._last_pan   = None
        self._last_tilt  = None
        self._laser_on   = False

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Float32, '/turret/pan_deg',  self._pan_cb,   10)
        self.create_subscription(Float32, '/turret/tilt_deg', self._tilt_cb,  10)
        self.create_subscription(Bool,    '/laser/fire',      self._laser_cb, 10)

        self.get_logger().info('TurretControllerNode ready.')

    # ──────────────────────────────────────────────────────────────────────────
    def _pan_cb(self, msg: Float32):
        angle = self._clamp(msg.data, 0.0, 180.0)
        if angle != self._last_pan:
            self._send(f'P{angle:.1f}')
            self._last_pan = angle

    def _tilt_cb(self, msg: Float32):
        angle = self._clamp(msg.data, 0.0, 180.0)
        if angle != self._last_tilt:
            self._send(f'T{angle:.1f}')
            self._last_tilt = angle

    def _laser_cb(self, msg: Bool):
        cmd = 'L1' if msg.data else 'L0'
        if msg.data != self._laser_on:
            self._send(cmd)
            self._laser_on = msg.data

    # ──────────────────────────────────────────────────────────────────────────
    def _send(self, cmd: str):
        """Send ASCII command to Arduino over serial."""
        line = cmd + '\n'
        if self._dry:
            self.get_logger().info(f'[DRY-RUN] serial → "{cmd}"')
        else:
            try:
                self._serial.write(line.encode('ascii'))
            except Exception as e:
                self.get_logger().error(f'Serial write error: {e}')

    @staticmethod
    def _clamp(val, lo, hi):
        return max(lo, min(hi, val))

    def destroy_node(self):
        if self._serial and self._serial.is_open:
            self._send('L0')  # laser off on shutdown
            self._serial.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TurretControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
PYEOF
log "turret_controller_node.py written"


# =============================================================================
# FILE 4: face_task.launch.py
# =============================================================================
log "Writing face_task.launch.py"
cat > src/perception/launch/face_task.launch.py << 'PYEOF'
#!/usr/bin/env python3
"""
face_task.launch.py
===================
Launch all three face recognition nodes together.

Usage:
    ros2 launch perception face_task.launch.py \
        target_image_path:=/path/to/target.jpg \
        dry_run:=false \
        serial_port:=/dev/ttyUSB1 \
        similarity_threshold:=0.35

To start the task after launching:
    ros2 topic pub /face_task/start std_msgs/Bool "data: true" --once

To monitor:
    ros2 topic echo /face/best_similarity
    ros2 topic echo /face_task/complete
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        # ── Launch arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument('target_image_path',    default_value='',
            description='Absolute path to target face image (JPG/PNG)'),

        DeclareLaunchArgument('similarity_threshold', default_value='0.35',
            description='ArcFace cosine similarity threshold (0.30-0.40 recommended)'),

        DeclareLaunchArgument('detection_size',       default_value='320',
            description='InsightFace detection input size (320 or 640)'),

        DeclareLaunchArgument('serial_port',          default_value='/dev/ttyUSB1',
            description='Arduino serial port'),

        DeclareLaunchArgument('dry_run',              default_value='true',
            description='dry_run=true: log commands only, no real serial/hardware'),

        DeclareLaunchArgument('settle_time_sec',      default_value='0.5',
            description='Seconds to wait after moving turret before capturing'),

        DeclareLaunchArgument('laser_on_time_sec',    default_value='3.0',
            description='Duration to keep laser on after match'),

        # ── Node 1: Vision ─────────────────────────────────────────────────────
        Node(
            package='perception',
            executable='face_recognition',
            name='face_recognition_node',
            output='screen',
            parameters=[{
                'target_image_path':    LaunchConfiguration('target_image_path'),
                'similarity_threshold': LaunchConfiguration('similarity_threshold'),
                'detection_size':       LaunchConfiguration('detection_size'),
            }]
        ),

        # ── Node 2: Task brain ─────────────────────────────────────────────────
        Node(
            package='perception',
            executable='face_task',
            name='face_task_node',
            output='screen',
            parameters=[{
                'settle_time_sec':   LaunchConfiguration('settle_time_sec'),
                'laser_on_time_sec': LaunchConfiguration('laser_on_time_sec'),
            }]
        ),

        # ── Node 3: Hardware bridge ────────────────────────────────────────────
        Node(
            package='perception',
            executable='turret_controller',
            name='turret_controller_node',
            output='screen',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'dry_run':     LaunchConfiguration('dry_run'),
            }]
        ),

    ])
PYEOF
log "face_task.launch.py written"


# =============================================================================
# FILE 5: Arduino firmware
# =============================================================================
log "Writing arduino/turret_bridge/turret_bridge.ino"
cat > arduino/turret_bridge/turret_bridge.ino << 'INOEOF'
/*
 * turret_bridge.ino
 * =================
 * Arduino firmware for the pan/tilt turret + laser.
 *
 * Hardware connections (ASSUMPTION - adjust to your wiring):
 *   Pin 9  --> Pan servo signal wire  (horizontal)
 *   Pin 10 --> Tilt servo signal wire (vertical)
 *   Pin 7  --> Laser module IN pin    (HIGH = ON)
 *   GND    --> Common ground for servos + laser
 *   5V     --> Servo VCC (or use external 5V/6V for high-torque servos)
 *
 * Serial protocol (115200 baud):
 *   P<float>\n  -- set pan  servo angle 0-180 degrees
 *   T<float>\n  -- set tilt servo angle 0-180 degrees
 *   L1\n        -- laser ON
 *   L0\n        -- laser OFF
 *
 * Example:
 *   "P90.0\n"  --> pan servo to 90 degrees (centre)
 *   "T65.0\n"  --> tilt servo to 65 degrees
 *   "L1\n"     --> laser on
 */

#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PIN_PAN   = 9;
const int PIN_TILT  = 10;
const int PIN_LASER = 7;

void setup() {
    Serial.begin(115200);
    panServo.attach(PIN_PAN);
    tiltServo.attach(PIN_TILT);
    pinMode(PIN_LASER, OUTPUT);
    digitalWrite(PIN_LASER, LOW);

    // Initialise to centre position
    panServo.write(90);
    tiltServo.write(90);

    Serial.println("READY");
}

void loop() {
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) return;

        char cmd = line.charAt(0);
        String val = line.substring(1);

        if (cmd == 'P') {
            float angle = val.toFloat();
            angle = constrain(angle, 0.0, 180.0);
            panServo.write((int)angle);
            Serial.print("PAN:");
            Serial.println(angle);

        } else if (cmd == 'T') {
            float angle = val.toFloat();
            angle = constrain(angle, 0.0, 180.0);
            tiltServo.write((int)angle);
            Serial.print("TILT:");
            Serial.println(angle);

        } else if (cmd == 'L') {
            if (val == "1") {
                digitalWrite(PIN_LASER, HIGH);
                Serial.println("LASER:ON");
            } else {
                digitalWrite(PIN_LASER, LOW);
                Serial.println("LASER:OFF");
            }
        }
    }
}
INOEOF
log "turret_bridge.ino written"


# =============================================================================
# FILE 6: Patch src/perception/setup.py
# Add three new console_scripts entries
# =============================================================================
log "Patching src/perception/setup.py"

# Check if already patched
if grep -q "face_recognition" src/perception/setup.py; then
    warn "setup.py already contains face_recognition entries — skipping patch"
else
    python3 - << 'PYEOF'
import re

with open('src/perception/setup.py', 'r') as f:
    content = f.read()

old = "        'console_scripts': [\n            'lane_costmap = perception.lane_costmap:main',\n            'calibrate_homography = perception.calibrate_homography:main',"
new = "        'console_scripts': [\n            'lane_costmap = perception.lane_costmap:main',\n            'calibrate_homography = perception.calibrate_homography:main',\n            'face_recognition = perception.face_recognition_node:main',\n            'face_task = perception.face_task_node:main',\n            'turret_controller = perception.turret_controller_node:main',"

if old in content:
    content = content.replace(old, new)
    with open('src/perception/setup.py', 'w') as f:
        f.write(content)
    print('setup.py patched successfully')
else:
    # Fallback: find the console_scripts block and append
    pattern = r"('console_scripts':\s*\[)(.*?)\]"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        new_entries = "\n            'face_recognition = perception.face_recognition_node:main',\n            'face_task = perception.face_task_node:main',\n            'turret_controller = perception.turret_controller_node:main',"
        replacement = match.group(1) + match.group(2).rstrip() + new_entries + "\n        ]"
        content = content[:match.start()] + replacement + content[match.end():]
        with open('src/perception/setup.py', 'w') as f:
            f.write(content)
        print('setup.py patched via regex fallback')
    else:
        print('WARNING: Could not auto-patch setup.py — add entries manually (see instructions)')
PYEOF
fi
log "setup.py patch done"


# =============================================================================
# FILE 7: Patch src/perception/package.xml
# Add cv_bridge depend if not present
# =============================================================================
log "Patching src/perception/package.xml"

if grep -q "cv_bridge" src/perception/package.xml; then
    warn "package.xml already has cv_bridge — skipping"
else
    python3 - << 'PYEOF'
with open('src/perception/package.xml', 'r') as f:
    content = f.read()

# Insert after <depend>geometry_msgs</depend>
old = '  <depend>geometry_msgs</depend>'
new = '  <depend>geometry_msgs</depend>\n  <depend>cv_bridge</depend>\n  <depend>python3-opencv</depend>'

if old in content:
    content = content.replace(old, new, 1)
    with open('src/perception/package.xml', 'w') as f:
        f.write(content)
    print('package.xml patched')
else:
    print('WARNING: Could not auto-patch package.xml — add cv_bridge manually')
PYEOF
fi
log "package.xml patch done"


# =============================================================================
# FILE 8: Patch Dockerfile
# Add insightface + onnxruntime + pyserial installs
# =============================================================================
log "Patching Dockerfile"

if grep -q "insightface" Dockerfile; then
    warn "Dockerfile already has insightface — skipping"
else
    python3 - << 'PYEOF'
with open('Dockerfile', 'r') as f:
    content = f.read()

# Insert pip installs before the CMD line
old = 'CMD ["/bin/bash"]'
new = ('# Face recognition dependencies\n'
       'RUN pip3 install insightface onnxruntime pyserial opencv-python-headless numpy --break-system-packages\n\n'
       'CMD ["/bin/bash"]')

if old in content:
    content = content.replace(old, new)
    with open('Dockerfile', 'w') as f:
        f.write(content)
    print('Dockerfile patched')
else:
    print('WARNING: Could not auto-patch Dockerfile CMD line — add pip install manually')
PYEOF
fi
log "Dockerfile patch done"


# =============================================================================
# Git add all new/modified files
# =============================================================================
log "Staging all changes with git add"
git add \
    src/perception/perception/face_recognition_node.py \
    src/perception/perception/face_task_node.py \
    src/perception/perception/turret_controller_node.py \
    src/perception/launch/face_task.launch.py \
    arduino/turret_bridge/turret_bridge.ino \
    src/perception/setup.py \
    src/perception/package.xml \
    Dockerfile

git commit -m "feat: add Plan 3 face recognition task (18-image grid, 2-servo turret)

- face_recognition_node.py: InsightFace ArcFace vision node
- face_task_node.py: state machine (IDLE→SCAN→FINE_TUNE→FIRE→DONE)
- turret_controller_node.py: serial bridge to Arduino for servos+laser
- face_task.launch.py: single launch file for all 3 nodes
- arduino/turret_bridge.ino: Arduino PWM firmware
- Patched setup.py, package.xml, Dockerfile"

# =============================================================================
# DONE — print summary
# =============================================================================
echo ""
echo "============================================================"
echo -e "${GREEN}  SETUP COMPLETE${NC}"
echo "============================================================"
echo ""
echo "Branch created: feature/face-recognition-plan3"
echo ""
echo "Files created:"
echo "  src/perception/perception/face_recognition_node.py"
echo "  src/perception/perception/face_task_node.py"
echo "  src/perception/perception/turret_controller_node.py"
echo "  src/perception/launch/face_task.launch.py"
echo "  arduino/turret_bridge/turret_bridge.ino"
echo ""
echo "Files modified:"
echo "  src/perception/setup.py"
echo "  src/perception/package.xml"
echo "  Dockerfile"
echo ""
echo "============================================================"
echo -e "${YELLOW}  NEXT STEPS${NC}"
echo "============================================================"
echo ""
echo "STEP 1 — Build the Docker image (picks up Dockerfile changes):"
echo "  sudo docker compose build"
echo ""
echo "STEP 2 — Enter the container:"
echo "  sudo docker compose run ros"
echo ""
echo "STEP 3 — Build the ROS2 workspace:"
echo "  cd /root/mercury"
echo "  colcon build --packages-select perception"
echo "  source install/setup.bash"
echo ""
echo "STEP 4 — Copy your target face image into the container:"
echo "  (from host) docker cp /path/to/target.jpg ros2_dev:/root/mercury/target.jpg"
echo ""
echo "STEP 5 — Launch all three nodes:"
echo "  ros2 launch perception face_task.launch.py \\"
echo "      target_image_path:=/root/mercury/target.jpg \\"
echo "      similarity_threshold:=0.35 \\"
echo "      dry_run:=true"
echo "  (dry_run:=true = no real hardware, just logs)"
echo ""
echo "STEP 6 — In a second terminal, start the task:"
echo "  docker exec -it ros2_dev bash"
echo "  source /root/mercury/install/setup.bash"
echo "  ros2 topic pub /face_task/start std_msgs/msg/Bool 'data: true' --once"
echo ""
echo "STEP 7 — Watch output:"
echo "  ros2 topic echo /face/best_similarity"
echo "  ros2 topic echo /face_task/complete"
echo ""
echo "STEP 8 — When ready for real hardware:"
echo "  1. Upload arduino/turret_bridge/turret_bridge.ino to your Arduino"
echo "  2. Connect Arduino to Jetson via USB"
echo "  3. Find port: ls /dev/ttyUSB*"
echo "  4. Relaunch with dry_run:=false serial_port:=/dev/ttyUSB1"
echo ""
echo "STEP 9 — Trigger from your navigation node:"
echo "  When robot reaches Waypoint 2, publish:"
echo "  ros2 topic pub /face_task/start std_msgs/msg/Bool 'data: true' --once"
echo ""
echo "============================================================"
echo -e "${YELLOW}  ASSUMPTIONS MADE${NC}"
echo "============================================================"
echo ""
echo "1. Servo control: Arduino Uno/Nano connected via USB serial"
echo "   Pan servo  → Arduino pin 9"
echo "   Tilt servo → Arduino pin 10"
echo "   Laser      → Arduino pin 7 (HIGH=ON)"
echo ""
echo "2. Serial port: /dev/ttyUSB1 (change serial_port:= at launch if different)"
echo "   (ttyUSB0 is likely your RPLidar A3)"
echo ""
echo "3. Both servos are standard 0-180° PWM servos"
echo ""
echo "4. Laser module accepts 3.3V/5V TTL HIGH signal to activate"
echo ""
echo "5. Vertical servo angles in degrees:"
echo "   Level 1 (mid)  = 40°  → covers roughly face height"
echo "   Level 2 (low)  = 25°  → looking slightly downward"
echo "   Level 3 (high) = 59°  → looking slightly upward"
echo "   YOU MUST CALIBRATE these for your actual robot height"
echo ""
echo "6. Horizontal scan: -75° to +75° in 30° steps from robot forward"
echo "   Pan servo centre (forward) = pan_centre_deg=90"
echo ""
echo "7. Target image is a clear frontal photo of the person's face"
echo "   provided before competition and placed at target_image_path"
echo ""
echo "8. dry_run=true by default — no hardware needed for initial testing"
echo ""
echo "9. onnxruntime (CPU) used in Dockerfile, not onnxruntime-gpu,"
echo "   to avoid CUDA dependency issues during build."
echo "   On Jetson with JetPack: replace with onnxruntime-gpu in Dockerfile"
echo ""
echo "10. Task is triggered externally via /face_task/start topic"
echo "    Your navigation stack must publish this when at Waypoint 2"
echo "============================================================"
