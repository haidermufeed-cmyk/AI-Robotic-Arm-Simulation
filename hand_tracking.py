
import cv2
import mediapipe as mp
import requests
import time
import math
import threading

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

# CAMERA
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera could not be opened!")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
frame_lock = threading.Lock()

preview_frame = None
preview_lock = threading.Lock()

running = True


def camera_reader():
    global latest_frame

    while running:
        success, frame = cap.read()

        if success:
            with frame_lock:
                latest_frame = frame


def preview_sender():
    global preview_frame

    sender = requests.Session()
    last_send = 0

    try:
        while running:
            now = time.perf_counter()

            if now - last_send < 0.066:
                time.sleep(0.002)
                continue

            with preview_lock:
                frame = preview_frame.copy() if preview_frame is not None else None

            if frame is None:
                continue

            frame = cv2.resize(frame, (320, 240))

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 50]
            )

            if ok:
                try:
                    sender.post(
                        "http://127.0.0.1:5000/video_frame",
                        data=encoded.tobytes(),
                        headers={"Content-Type": "image/jpeg"},
                        timeout=0.5
                    )
                except requests.RequestException:
                    pass

            last_send = time.perf_counter()

    finally:
        sender.close()


camera_thread = threading.Thread(target=camera_reader, daemon=True)
preview_thread = threading.Thread(target=preview_sender, daemon=True)

camera_thread.start()
preview_thread.start()

# MEDIAPIPE
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

server_url = "http://127.0.0.1:5000/hand"
session = requests.Session()

last_send_time = 0
last_detected = False

print("Palm tracking started. Camera is displayed in the dashboard.")
print("Press CTRL+C in this terminal to stop.")

try:
    while True:

        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        current_time = time.perf_counter()
        detected = bool(results.multi_hand_landmarks)

        h, w, _ = frame.shape
        pinch = False

        if detected:
            landmarks = results.multi_hand_landmarks[0]

            palm_points = [0, 5, 9, 13, 17]

            palm_x = sum(
                landmarks.landmark[i].x for i in palm_points
            ) / 5

            palm_y = sum(
                landmarks.landmark[i].y for i in palm_points
            ) / 5

            x = int(palm_x * w)
            y = int(palm_y * h)

            index_tip = landmarks.landmark[8]
            thumb_tip = landmarks.landmark[4]

            distance = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            pinch = distance < 0.055

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            cv2.putText(
                frame,
                "GRIP CLOSED" if pinch else "GRIP OPEN",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            if current_time - last_send_time >= 0.05:
                try:
                    session.post(
                        server_url,
                        json={
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "detected": True,
                            "pinch": pinch
                        },
                        timeout=0.2
                    )
                except requests.RequestException:
                    pass

                last_send_time = current_time

            last_detected = True

        else:
            cv2.putText(
                frame,
                "Show your hand",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            if last_detected:
                try:
                    session.post(
                        server_url,
                        json={
                            "x": 0,
                            "y": 0,
                            "width": w,
                            "height": h,
                            "detected": False,
                            "pinch": False
                        },
                        timeout=0.2
                    )
                except requests.RequestException:
                    pass

                last_detected = False

        with preview_lock:
            preview_frame = frame.copy()

finally:
    running = False

    camera_thread.join(timeout=2)
    preview_thread.join(timeout=2)

    cap.release()
    hands.close()
    session.close()
