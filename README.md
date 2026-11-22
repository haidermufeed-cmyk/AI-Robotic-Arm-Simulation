# AI Robotic Arm Simulation

A vision-based robotic arm simulation controlled through real-time hand gestures using computer vision.

## 🎥 Project Demonstration

**Watch the robotic arm in action:**

[▶ View Demo Video (MP4)](demo.mp4)

Download or open the video to see real-time hand tracking and robotic arm movement.

## Overview

This project combines computer vision, hand tracking, Python, and 3D visualization to create an interactive robotic arm simulation.

Users can control the simulated robotic arm through hand movements captured by a webcam.

## Features

- Real-time hand tracking using MediaPipe
- Webcam-based gesture recognition
- Palm movement-based robotic arm control
- Thumb and index finger pinch detection
- Interactive gripper control
- 3D robotic arm visualization
- Live camera feed in the dashboard
- Real-time communication between Python and the web dashboard
- One-click project launch through VS Code

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main application logic |
| OpenCV | Webcam and image processing |
| MediaPipe | Hand landmark detection |
| Flask | Backend communication |
| HTML, CSS, JavaScript | Dashboard interface |
| Three.js | 3D robotic arm visualization |
| NumPy | Numerical operations |

## Project Structure

```text
Robotic_arm/
├── .vscode/
│   └── tasks.json
├── app.py
├── hand_tracking.py
├── index.html
├── demo.mp4
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/haidermufeed-cmyk/AI-Robotic-Arm-Simulation.git
```

### 2. Open the project folder

```bash
cd AI-Robotic-Arm-Simulation
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install opencv-python mediapipe==0.10.21 numpy flask flask-cors requests
```

## Running the Project

Open the project in Visual Studio Code.

Press:

```text
Ctrl + Shift + B
```

Select **Launch Everything**.

The application starts the backend, hand tracking, dashboard server, and browser.

Dashboard:

```text
http://localhost:8000
```

## How It Works

1. The webcam captures the user's hand.
2. MediaPipe detects hand landmarks.
3. Python processes palm position and pinch gestures.
4. Gesture data is sent to the Flask backend.
5. The web dashboard receives the data.
6. Three.js updates the simulated robotic arm.

## Future Improvements

- More advanced gesture recognition
- Improved robotic arm movement accuracy
- Multiple gesture-based commands
- Object interaction improvements
- Hardware integration with a physical robotic arm

## Developer

**Haider Mufeed**

GitHub: [haidermufeed-cmyk](https://github.com/haidermufeed-cmyk)

---

*An interactive exploration of computer vision, gesture control, and 3D robotics.*
