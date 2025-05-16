# 🌀 SynesthesIA

**A real-time gesture-to-visual-and-sound generator powered by Python, Blender, and AI.**  
A creative coding project transforming full-body movements into programmable audiovisual expressions.

---

## ⚙️ Current Status — Stage 1: Real-Time Pose Detection & Gesture Recognition

SynesthesIA currently captures full-body movement using a webcam, processes it via [MediaPipe Pose](https://github.com/google/mediapipe), and detects high-level body gestures using a custom `GestureRecognizer`. This forms the foundation for real-time interaction with 3D environments and audio synthesis.

### ✅ Implemented Features

- **Live webcam capture**
- **Pose estimation using MediaPipe**
- **Custom gesture recognition system**, with:
  - `power_up` (arms raised)
  - `crossed_arms`
  - `t_pose`
- **Heads-Up Display (HUD):**
  - Toggle gesture help overlay via `H`
  - Toggle scrollable gesture event log via `F4` (disabled by default)
- **Event log system** with timestamped gesture entries
- **Modular architecture** for extending gesture logic or connecting to external systems (e.g., Blender, OSC, UDP)

---

## 🧭 Roadmap

| Stage | Description |
|-------|-------------|
| **1. Tracking Core** | Real-time gesture detection pipeline using Python and MediaPipe ✅ |
| **2. Blender Integration** | Stream gestures via UDP to a custom Blender scene |
| **3. Audiovisual Mapping** | Map gestures to shaders + generative audio |
| **4. Artistic Reveal** | Polish visuals, add cinematic camera, stage a full live-coded experience |

---

## 🛠️ Setup

### Requirements

- Python 3.9+
- `opencv-python`
- `mediapipe`

Install dependencies:

```bash
pip install -r requirements.txt
