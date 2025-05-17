# SynesthesIA - Phase 1: Motion Tracking Prototype

**Real-time body gesture detection system**

## Current Implementation Status (Phase 1)

✅ **Core Functionality**
- Real-time body tracking via MediaPipe
- Basic gesture recognition (5 core gestures)
- Camera calibration system
- Event logging interface

🛠️ **In Development**
- OSC communication framework (partial)
- Basic audio triggering (stub implementation)
- Debug visualization tools

## System Architecture

```
src/
├── main.py                # Entry point
├── requirements.txt       # Dependencies
│
├───audio
│   └── audio_manager.py   # Audio output stub
│
├───communication
│   └── osc_manager.py     # OSC communication handler
│
├───tracking
│   ├── body_tracker.py    # Main tracking logic
│   └── gesture_recognition.py  # Gesture detection
│
└───visuals               # (Planned - currently in main)
```

## Detected Gestures (Phase 1)

| Gesture | Type | Detection Method |
|---------|------|------------------|
| Arms Raised | `POWER_UP` | Wrist-shoulder ratio |
| Arms Crossed | `CROSSED_ARMS` | Wrist proximity |
| T-Pose | `T_POSE` | Arm extension ratio |
| Left Arm Tap | `TAP_LEFT` | Velocity + proximity |
| Right Arm Tap | `TAP_RIGHT` | Velocity + proximity |

## Installation

1. Clone repository:
```bash
git clone https://github.com/NathanKneT/SynesthesIA.git
cd SynesthesIA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the System

Basic operation:
```bash
python src/main.py
```

## Keyboard Controls

| Key | Function |
|-----|----------|
| `H` | Toggle help display |
| `L` | Show event log |
| `D` | Toggle debug info |
| `C` | Start calibration |
| `ESC` | Quit application |

## Technical Details

**Body Tracking:**
- Uses MediaPipe Pose with complexity level 1
- Processes at 30 FPS on 720p input
- Adaptive thresholds based on user calibration

**Gesture Recognition:**
```python
# Sample detection logic
def detect_power_up(landmarks):
    left_raised = landmarks[LEFT_WRIST].y < landmarks[LEFT_SHOULDER].y - threshold
    right_raised = landmarks[RIGHT_WRIST].y < landmarks[RIGHT_SHOULDER].y - threshold
    return left_raised and right_raised
```

## Next Steps (Phase 2 Preview)

Planned features for audio integration:
- OSC audio triggering framework
- Basic sound synthesis engine
- Instrument mapping system

## Troubleshooting

**Common Issues:**
- **Camera not detected**: Try different camera indices (0, 1, 2)
- **High CPU usage**: Reduce camera resolution in `body_tracker.py`
- **False detections**: Run calibration (`C` key) in neutral position

## License

MIT License - See [LICENSE](LICENSE) for details.


## Dependencies

- [MediaPipe](https://google.github.io/mediapipe/) pour la détection de pose
- [OpenCV](https://opencv.org/) pour le traitement d'image
- [python-osc](https://github.com/attwad/python-osc) pour la communication OSC
- [sounddevice](https://python-sounddevice.readthedocs.io/) pour la génération audio