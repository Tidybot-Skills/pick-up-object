# pick-up-object

IBVS (Image-Based Visual Servoing) skill to pick up objects using wrist camera with mask centroid tracking.

Author: evilsky, jarvis  
Dependencies: none

## How it works

1. **Mask Centroid Tracking**: Uses YOLO segmentation masks for accurate centering
2. **Two-Phase Servoing**:
   - **Base frame** (Z > -0.35m): target = image center
   - **EE frame** (Z < -0.35m): target = gripper offset, EE points straight down
3. **Search on Detection Miss**: 
   - Sweeps ±5cm in X and Y
   - Rotates ±30° 
   - **Stays at found position** (doesn't return to center)
4. **Straight-Down Approach**: Sets EE orientation to (roll=π, pitch=0, yaw=0) for final descent
5. **Floor Contact Detection**: Descends until ArmError (convergence timeout)

## Key Features

- **Stay at found position**: When XY/rotation search finds the object, arm stays there
- **No yaw rotation during descent**: Last joint stays at 0°
- **Robust to spotty detection**: Searches aggressively, fails only after 3 consecutive search failures

## Usage

```python
from main import pick_up_object

success = pick_up_object("banana")
success = pick_up_object("apple")
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECTION_CONFIDENCE` | 0.15 | YOLO confidence threshold |
| `SEARCH_XY_STEP_M` | 0.05m | XY sweep distance |
| `SEARCH_WIGGLE_ANGLE_DEG` | 30° | Rotation search angle |
| `MAX_SEARCH_FAILURES` | 3 | Consecutive failures before abort |
| `EE_FRAME_Z_THRESHOLD` | -0.35m | Switch to EE frame below this Z |
| `GRIPPER_V_OFFSET` | -120px | Gripper target offset (EE frame only) |

## Requirements

- Tidybot robot with Franka Panda arm
- Wrist camera (RealSense)
- robot_sdk: arm, gripper, sensors, yolo, display

## Tested

- 2026-02-13: Search+stay + straight-down approach working (78% success rate)
