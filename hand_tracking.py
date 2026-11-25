
import cv2
import mediapipe as mp
import requests
import time
import math
import threading

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

SERVER = "http://127.0.0.1:5000"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened!")
    raise SystemExit

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera:", int(cap.get(3)), "x", int(cap.get(4)))


class OneEuroFilter:
    def __init__(self, min_cutoff=1.2, beta=0.045, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.value = None
        self.derivative = 0.0

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 0.001))

    def update(self, value, dt):
        if self.value is None:
            self.value = value
            return value

        raw_derivative = (value - self.value) / max(dt, 0.001)
        da = self.alpha(self.d_cutoff, dt)

        self.derivative += da * (raw_derivative - self.derivative)

        cutoff = self.min_cutoff + self.beta * abs(self.derivative)
        va = self.alpha(cutoff, dt)

        self.value += va * (value - self.value)
        return self.value


landmark_filters = [
    [OneEuroFilter() for _ in range(3)]
    for _ in range(21)
]


def smooth_landmarks(landmarks, dt):
    output = []

    for i, point in enumerate(landmarks):
        x = landmark_filters[i][0].update(point.x, dt)
        y = landmark_filters[i][1].update(point.y, dt)
        z = landmark_filters[i][2].update(point.z, dt)
        output.append((x, y, z))

    return output


def reset_landmark_filters():
    global landmark_filters
    landmark_filters = [
        [OneEuroFilter() for _ in range(3)]
        for _ in range(21)
    ]


latest_frame = None
frame_id = 0
frame_lock = threading.Lock()

preview_frame = None
preview_lock = threading.Lock()

latest_hand_data = {
    "x": 0,
    "y": 0,
    "width": 320,
    "height": 240,
    "detected": False,
    "pinch": False,
    "elbow": 0.5,
    "wrist": 0.0
}

hand_data_lock = threading.Lock()
running = True

camera_fps = 0
tracking_fps = 0
preview_fps = 0
stats_lock = threading.Lock()


def camera_reader():
    global latest_frame, frame_id, camera_fps

    count = 0
    start = time.perf_counter()

    while running:
        success, frame = cap.read()

        if not success:
            time.sleep(0.005)
            continue

        with frame_lock:
            latest_frame = frame
            frame_id += 1

        count += 1
        elapsed = time.perf_counter() - start

        if elapsed >= 1:
            with stats_lock:
                camera_fps = count / elapsed

            count = 0
            start = time.perf_counter()


def hand_sender():
    sender = requests.Session()

    while running:
        with hand_data_lock:
            data = latest_hand_data.copy()

        try:
            response = sender.post(
                SERVER + "/hand",
                json=data,
                timeout=(0.3, 0.5)
            )
            response.close()
        except requests.RequestException:
            pass

        time.sleep(0.05)

    sender.close()


def preview_sender():
    global preview_fps

    sender = requests.Session()
    count = 0
    start = time.perf_counter()
    last_send = 0

    while running:
        with preview_lock:
            frame = preview_frame

        now = time.perf_counter()

        if frame is None:
            time.sleep(0.002)
            continue

        if now - last_send < 0.04:
            time.sleep(0.002)
            continue

        last_send = now

        small = cv2.resize(
            frame,
            (240, 180),
            interpolation=cv2.INTER_AREA
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            small,
            [cv2.IMWRITE_JPEG_QUALITY, 30]
        )

        if ok:
            try:
                response = sender.post(
                    SERVER + "/video_frame",
                    data=encoded.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                    timeout=(0.3, 0.5)
                )

                if response.status_code == 200:
                    count += 1

                response.close()

            except requests.RequestException:
                pass

        elapsed = time.perf_counter() - start

        if elapsed >= 1:
            with stats_lock:
                preview_fps = count / elapsed

            count = 0
            start = time.perf_counter()

    sender.close()


camera_thread = threading.Thread(target=camera_reader, daemon=True)
hand_thread = threading.Thread(target=hand_sender, daemon=True)
preview_thread = threading.Thread(target=preview_sender, daemon=True)

camera_thread.start()
hand_thread.start()
preview_thread.start()


mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


last_frame_id = -1
last_frame_time = time.perf_counter()

smooth_x = None
smooth_y = None
last_detected = None

tracking_count = 0
tracking_start = time.perf_counter()
last_report = time.perf_counter()

pinch_state = False
pinch_close_frames = 0
pinch_open_frames = 0
smoothed_pinch_ratio = None

PINCH_CLOSE_THRESHOLD = 0.42
PINCH_OPEN_THRESHOLD = 0.58

PINCH_CONFIRM_FRAMES = 3
PINCH_SMOOTHING = 0.45

print("\nROBOT ARM TRACKING - STABLE GRIP MODE")
print("Move hand: Base and shoulder")
print("Open/close fingers: Elbow signal")
print("Tilt hand: Wrist signal")
print("Pinch: Improved grip detection")
print("Brief tracking loss will preserve grip state")
print("Press CTRL+C to stop.\n")


try:
    while True:
        with frame_lock:
            current_id = frame_id
            frame = latest_frame

        if frame is None or current_id == last_frame_id:
            time.sleep(0.001)
            continue

        last_frame_id = current_id

        now = time.perf_counter()
        dt = min(now - last_frame_time, 0.1)
        last_frame_time = now

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = hands.process(rgb)

        h, w = frame.shape[:2]
        detected = bool(results.multi_hand_landmarks)

        pinch = pinch_state
        elbow = 0.5
        wrist = 0.0
        pinch_ratio = 0.0

        if detected:
            raw_landmarks = results.multi_hand_landmarks[0].landmark
            landmarks = smooth_landmarks(raw_landmarks, dt)

            palm_points = [0, 5, 9, 13, 17]

            palm_x = sum(landmarks[i][0] for i in palm_points) / 5
            palm_y = sum(landmarks[i][1] for i in palm_points) / 5

            raw_x = palm_x * w
            raw_y = palm_y * h

            if smooth_x is None:
                smooth_x = raw_x
                smooth_y = raw_y
            else:
                alpha = 0.45
                smooth_x += alpha * (raw_x - smooth_x)
                smooth_y += alpha * (raw_y - smooth_y)

            x = int(smooth_x)
            y = int(smooth_y)

            wrist_point = landmarks[0]
            middle_mcp = landmarks[9]

            palm_length = math.hypot(
                middle_mcp[0] - wrist_point[0],
                middle_mcp[1] - wrist_point[1]
            )

            if palm_length > 0.001:
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]

                finger_distance = math.hypot(
                    thumb_tip[0] - index_tip[0],
                    thumb_tip[1] - index_tip[1]
                )

                raw_ratio = finger_distance / palm_length

                if smoothed_pinch_ratio is None:
                    smoothed_pinch_ratio = raw_ratio
                else:
                    smoothed_pinch_ratio += PINCH_SMOOTHING * (
                        raw_ratio - smoothed_pinch_ratio
                    )

                pinch_ratio = smoothed_pinch_ratio

                if not pinch_state:
                    if pinch_ratio < PINCH_CLOSE_THRESHOLD:
                        pinch_close_frames += 1
                    else:
                        pinch_close_frames = 0

                    pinch_open_frames = 0

                    if pinch_close_frames >= PINCH_CONFIRM_FRAMES:
                        pinch_state = True
                        pinch_close_frames = 0
                        print("GRIP CLOSED")

                else:
                    if pinch_ratio > PINCH_OPEN_THRESHOLD:
                        pinch_open_frames += 1
                    else:
                        pinch_open_frames = 0

                    pinch_close_frames = 0

                    if pinch_open_frames >= PINCH_CONFIRM_FRAMES:
                        pinch_state = False
                        pinch_open_frames = 0
                        print("GRIP OPEN")

            pinch = pinch_state

            tips = [8, 12, 16, 20]

            average_extension = sum(
                math.hypot(
                    landmarks[i][0] - wrist_point[0],
                    landmarks[i][1] - wrist_point[1]
                )
                for i in tips
            ) / len(tips)

            openness = average_extension / max(palm_length, 0.001)

            elbow = max(
                0.0,
                min(1.0, (openness - 1.35) / 0.85)
            )

            index_mcp = landmarks[5]
            pinky_mcp = landmarks[17]

            tilt_angle = math.degrees(math.atan2(
                pinky_mcp[1] - index_mcp[1],
                pinky_mcp[0] - index_mcp[0]
            ))

            wrist = max(-1.0, min(1.0, tilt_angle / 45.0))

            thumb_xy = (
                int(landmarks[4][0] * w),
                int(landmarks[4][1] * h)
            )

            index_xy = (
                int(landmarks[8][0] * w),
                int(landmarks[8][1] * h)
            )

            cv2.circle(frame, thumb_xy, 6, (255, 0, 255), -1)
            cv2.circle(frame, index_xy, 6, (0, 255, 255), -1)
            cv2.line(frame, thumb_xy, index_xy, (255, 255, 255), 2)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

            grip_text = "GRIP CLOSED" if pinch else "GRIP OPEN"
            grip_color = (0, 165, 255) if pinch else (0, 255, 0)

            cv2.putText(frame, grip_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, grip_color, 2)

            cv2.putText(frame, f"PINCH: {pinch_ratio:.2f}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            cv2.putText(frame, f"ELBOW: {elbow:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            with hand_data_lock:
                latest_hand_data = {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "detected": True,
                    "pinch": pinch,
                    "elbow": elbow,
                    "wrist": wrist
                }

            if detected != last_detected:
                print("HAND FOUND")
                last_detected = detected

        else:
            smooth_x = None
            smooth_y = None

            reset_landmark_filters()

            # Keep the previous pinch state during brief tracking loss.
            pinch_close_frames = 0
            pinch_open_frames = 0

            pinch = pinch_state

            cv2.putText(frame, "Show your hand", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            with hand_data_lock:
                latest_hand_data = {
                    "x": 0,
                    "y": 0,
                    "width": w,
                    "height": h,
                    "detected": False,
                    "pinch": pinch,
                    "elbow": 0.5,
                    "wrist": 0.0
                }

            if detected != last_detected:
                print("NO HAND FOUND")
                last_detected = detected

        with preview_lock:
            preview_frame = frame

        tracking_count += 1
        elapsed = time.perf_counter() - tracking_start

        if elapsed >= 1:
            with stats_lock:
                tracking_fps = tracking_count / elapsed

            tracking_count = 0
            tracking_start = time.perf_counter()

        if time.perf_counter() - last_report >= 2:
            with stats_lock:
                cam = camera_fps
                track = tracking_fps
                preview = preview_fps

            print(
                f"Camera: {cam:.1f} FPS | "
                f"MediaPipe: {track:.1f} FPS | "
                f"Preview: {preview:.1f} FPS"
            )

            last_report = time.perf_counter()

except KeyboardInterrupt:
    print("\nStopping camera...")

finally:
    running = False

    camera_thread.join(timeout=2)
    hand_thread.join(timeout=2)
    preview_thread.join(timeout=2)

    cap.release()
    hands.close()

    print("Camera stopped.")
