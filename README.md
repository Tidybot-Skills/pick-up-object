# pick-up-object

IBVS (Image-Based Visual Servoing) skill to pick up objects from the ground using wrist camera.

## How it works

1. **Visual Servoing**: Uses YOLO detection on wrist camera to center the target object in the camera frame
2. **Adaptive Gains**: Faster corrections when far, finer control when close
3. **Two-Stage Descent**: Fast drop (70%) then slow final approach (30%) to ground level
4. **Grasp & Lift**: Close gripper with force, then lift

## Parameters

- `GROUND_Z = -0.65` - Ground level in arm base frame
- `CENTER_THRESH = 50px` - Pixel threshold for "centered"
- `BBOX_READY = 130px` - Bbox size threshold for "close enough"
- Adaptive gains: 0.00025 (far), 0.00018 (mid), 0.00012 (close)

## Usage

```python
from main import pick_up_object

# Pick up a banana from the ground
success = pick_up_object("banana")

# Pick up any YOLO-detectable object
success = pick_up_object("apple")
```

## Requirements

- Tidybot robot with Franka arm
- Wrist camera (RealSense)
- robot_sdk: yolo, arm, gripper, sensors

## Testing

Tested with banana on ground (2026-02-10):
- IBVS converged in ~60 iterations
- Lowest z reached: -0.644 (arm can reach ground!)
- Successful grasp and lift

## Notes

- The arm starts at z≈0.24 and descends to z≈-0.65 (total ~0.9m drop)
- IBVS handles both X and Y centering before descent
- Two-stage descent prevents collision with ground
