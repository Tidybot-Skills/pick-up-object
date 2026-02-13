# pick-up-object

IBVS (Image-Based Visual Servoing) skill to pick up objects using wrist camera with mask centroid tracking.

Author: evilsky, jarvis
Dependencies: none

## How it works

1. **Mask Centroid Tracking**: Uses YOLO segmentation masks (not just bbox) for sub-pixel accurate centering
2. **Two-Phase Servoing**:
   - **Base frame** (coarse): Above Z=-0.35m, uses base frame gains for rough approach
   - **EE frame** (fine): Below Z=-0.35m, switches to end-effector frame with calibrated gains
3. **Search Wiggle**: On detection miss, rotates last joint ±30° to find object from different angles
4. **Yaw Alignment**: Computes object major axis from mask covariance, rotates gripper perpendicular for optimal grasp
5. **Simultaneous Descent**: Centers AND descends each iteration (pauses descent if centering error too large)
6. **Floor Contact Detection**: Descends until ArmError (convergence timeout = floor contact)

## Key Features

- **Gripper offset compensation**: Targets upper portion of camera frame where gripper actually is
- **Auto-calibration routine**: `auto_calibrate()` determines correct gain signs/magnitudes
- **Elongation-aware rotation**: Only rotates gripper for elongated objects (ratio > 2.0)
- **Lost object recovery**: Retries detection up to 3 consecutive misses before aborting

## Usage

```python
from main import pick_up_object

# Pick up any YOLO-detectable object
success = pick_up_object("banana")
success = pick_up_object("apple")
success = pick_up_object("pen")

# Run auto-calibration (once per setup)
from main import auto_calibrate
auto_calibrate("banana")  # Prints recommended gains
```

## Configuration

Key parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_ID` | wrist_cam | Camera to use for detection |
| `DETECTION_CONFIDENCE` | 0.15 | YOLO confidence threshold |
| `GRIPPER_V_OFFSET` | -120 px | Vertical offset for gripper alignment |
| `EE_FRAME_Z_THRESHOLD` | -0.35m | Z height to switch to EE frame |
| `DESCEND_STEP_M` | 0.05m | Descent per iteration |
| `DESCEND_PAUSE_PIXELS` | 80 px | Pause descent if error exceeds this |
| `SEARCH_WIGGLE_ANGLE_DEG` | 30° | Angle to rotate when searching |
| `MAX_SEARCH_FAILURES` | 10 | Max consecutive failed searches before abort |

## Requirements

- Tidybot robot with Franka Panda arm
- Wrist camera (RealSense 309622300814)
- robot_sdk: arm, gripper, sensors, yolo, display

## Tested

- 2026-02-12: Mask centroid + EE frame servoing working
- Yaw alignment tracks elongated objects
- Floor contact detection reliable
