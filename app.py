
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from threading import Lock, Condition

app = Flask(__name__)
CORS(app)

hand_position = {
    "x": 0,
    "y": 0,
    "width": 320,
    "height": 240,
    "detected": False,
    "pinch": False
}

position_lock = Lock()

video_condition = Condition()
latest_frame = None
video_version = 0


@app.route("/hand", methods=["POST"])
def update_hand():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"status": "invalid data"}), 400

    with position_lock:
        hand_position.update(data)

    return jsonify({"status": "success"})


@app.route("/hand", methods=["GET"])
def get_hand():
    with position_lock:
        return jsonify(hand_position)


@app.route("/video_frame", methods=["POST"])
def receive_video():
    global latest_frame, video_version

    data = request.get_data()

    if not data:
        return jsonify({"status": "empty frame"}), 400

    with video_condition:
        latest_frame = data
        video_version += 1
        video_condition.notify_all()

    return jsonify({"status": "received"})


@app.route("/video_feed")
def video_feed():

    def generate():
        last_version = -1

        while True:
            with video_condition:
                video_condition.wait_for(
                    lambda: video_version != last_version,
                    timeout=10
                )

                if latest_frame is None:
                    continue

                frame = latest_frame
                last_version = video_version

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + frame
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
