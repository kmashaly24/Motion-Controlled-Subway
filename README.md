# Motion-Controlled Subway Surfers

Play Subway Surfers using your body instead of a keyboard or controller.

A webcam and MediaPipe pose tracking detect your movements in real time and translate them into in-game key presses.

## Features

- Real-time body tracking with MediaPipe Pose
- Jump and duck detection from head movement
- 3-lane control using shoulder position
- Auto-pause when you stop moving or leave frame
- Automatic calibration based on your height and camera distance
- Simulation mode for testing without sending keys

## Requirements

- Windows 10/11
- Python 3.12
- A webcam
- Subway Surfers running in focus

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_env.py
```

## Run

```bash
python subway_surfers_controller.py
```

The script starts with a short calibration phase:

1. Stand in your normal pose for the first countdown.
2. Wait for the second countdown before focusing the game window.
3. Play by moving your body.

Keep the game window focused so keystrokes reach Subway Surfers.

## Controls

- Move left/right shoulder position to change lanes
- Raise your head to jump
- Lower your head to duck
- Stay still or leave the frame to trigger auto-pause

Press `q` in the video window to quit.

## How it works

The controller uses MediaPipe Pose to track key landmarks such as the nose and shoulders. It then:

- measures vertical motion to detect jump/duck gestures
- uses the shoulder midpoint to decide the active lane
- smooths movement and applies pause/resume logic
- sends low-latency key presses to the game using PyAutoGUI

## Simulation mode

To run without sending real keyboard input, set this in the script:

```python
ENABLE_GAME_CONTROL = False
```

This keeps the webcam and detection pipeline active while printing detected actions instead of pressing keys.

## Notes

- Best performance is achieved with good lighting and a clear view of your upper body.
- The system is designed to adapt to your height and camera distance automatically.
- The game window should remain in focus while playing.


