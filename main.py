"""Pick object with image-based visual servoing (IBVS) — 2D only.

Continuously centers the target object in the camera image while descending
toward it, then grasps it. Uses YOLO 2D segmentation with mask centroid
for sub-bbox-level accuracy.

Workflow:
  1. Detect target with YOLO 2D (with mask) → get pixel center
  2. Servo-descend loop: at every step, detect → correct lateral error AND
     descend a small increment. Keeps object centered throughout approach.
  3. On detection miss: wiggle last joint ±30° AND sweep XY ±5cm to search
  4. Grasp after descending to floor contact
  5. Lift and return home

Usage (submit via code execution API):
  curl -X POST localhost:8080/code/execute \
    -H "X-Lease-Id: <lease>" \
    -H "Content-Type: application/json" \
    -d '{"code": "exec(open(\"pick_up_object.py\").read())"}'

Note:
  The camera-to-EE frame mapping depends on how the camera is mounted.
  Tune the GAIN_* values below for your setup. The auto_calibrate() routine
  can determine the correct signs and rough magnitudes.
"""

from robot_sdk import arm, gripper, sensors, yolo, display
from robot_sdk.arm import ArmError
import numpy as np
import time
import math

# ============================================================================
# Configuration — tune these for your setup
# ============================================================================

# Target object to detect (YOLO text prompt)
TARGET_OBJECT = "banana"

# Camera to use (wrist_cam sees the floor/objects, not base_cam)
CAMERA_ID = "309622300814"  # wrist_cam

# YOLO detection confidence threshold
DETECTION_CONFIDENCE = 0.15

# --- Visual servoing gains ---
# Maps pixel error (u_err, v_err) to EE delta in base frame (dx, dy, dz).
#
# Convention:
#   u_err = u_object - u_center  (positive = object is right of center)
#   v_err = v_object - v_center  (positive = object is below center)
#
# Gain units: meters per pixel. Typical values: 0.0001 to 0.001 m/px.
# Sign determines direction — flip if the servoing diverges.

# Pixel error in u (horizontal) maps to EE delta:
GAIN_U_TO_DY = -0.0006  # u_err → dy (flipped: object right → move EE right)
GAIN_U_TO_DX = 0.0      # u_err → dx (usually 0 for downward-looking camera)

# Pixel error in v (vertical) maps to EE delta:
GAIN_V_TO_DX = -0.0006  # v_err → dx (flipped: object below → move EE forward)
GAIN_V_TO_DZ = 0.0      # v_err → dz (usually 0 for downward-looking camera)

# --- Gripper-to-camera offset ---
# The gripper center is at the top portion of the wrist camera image,
# not at the image center. These offsets shift the servo target point
# so the object ends up between the gripper fingers.
# Negative GRIPPER_V_OFFSET = target is above image center (toward top of frame).
# NOTE: Only used in EE frame phase. Base frame phase uses image center.
GRIPPER_U_OFFSET = 0.0     # Horizontal offset in pixels (0 = centered)
GRIPPER_V_OFFSET = -120    # Vertical offset in pixels (negative = upper portion)

# --- Servoing loop parameters ---
PIXEL_TOLERANCE = 30       # Centering tolerance in pixels
MAX_SERVO_ITERATIONS = 200 # Maximum total servo-descend iterations
MAX_LATERAL_STEP_M = 0.05  # Clamp each lateral EE step (meters)
MIN_LATERAL_STEP_M = 0.001 # Ignore lateral steps smaller than this (meters)
SERVO_MOVE_DURATION = 0.5  # Duration for each small move (seconds)

# --- Search wiggle parameters ---
# When detection fails, rotate last joint AND sweep XY to search for object
SEARCH_WIGGLE_ANGLE_DEG = 30  # Degrees to rotate in each direction
SEARCH_XY_STEP_M = 0.05       # Meters to sweep in X and Y (±5cm)
SEARCH_WIGGLE_DURATION = 0.4  # Duration for wiggle move (seconds)
MAX_SEARCH_FAILURES = 3       # Max consecutive search failures before giving up

# --- Descent parameters ---
# No fixed floor height — descend until ArmError (convergence timeout),
# which means the arm hit the floor and can't reach the commanded position.
DESCEND_STEP_M = 0.05        # Descent step per iteration (meters)
DESCEND_PAUSE_PIXELS = 80    # Pause descent if pixel error exceeds this

# --- EE frame mode (close range) ---
# Below this Z threshold, switch from base frame to EE frame servoing.
# EE frame gains from calibration (see ee_frame_calibration_results.md):
#   image U ≈ -EE_Y  (~580 px/m)
#   image V ≈ +EE_X  (~660 px/m)
EE_FRAME_Z_THRESHOLD = -0.35  # Switch to EE frame below this Z (meters)
EE_GAIN_U_TO_DY = +1.0 / 580  # u_err → dy_ee (≈ +0.0017 m/px): object right → +EE_Y → pushes object left
EE_GAIN_V_TO_DX = -1.0 / 660  # v_err → dx_ee (≈ -0.0015 m/px): object below → -EE_X → pushes object up
EE_YAW_GAIN = 0.5             # Fraction of yaw error to correct per step
EE_MIN_ELONGATION = 2.0       # Only rotate if elongation ratio exceeds this

# --- Grasp parameters ---
LIFT_TARGET_Z = 0.05       # Absolute Z to lift to after grasping (above 0.0 in arm base frame)
GRASP_FORCE = 50           # Gripper grasp force (0-255)
GRASP_SPEED = 200          # Gripper grasp speed (0-255)


# ============================================================================
# Helper functions
# ============================================================================

def detect_object_2d(target: str, confidence: float = DETECTION_CONFIDENCE):
    """Detect target object with mask and return the best detection.

    Requests masks so we can compute the mask centroid for more accurate
    centering than the bounding box center alone.

    Returns:
        (Detection, image_shape) or (None, None) if not found.
    """
    result = yolo.segment_camera(
        target, camera_id=CAMERA_ID, confidence=confidence,
        save_visualization=True, mask_format="npz",
    )
    detections = result.get_by_class(target)
    if not detections:
        detections = result.detections
    if not detections:
        return None, None
    # Pick the largest detection
    best = max(detections, key=lambda d: d.area if d.area > 0 else
               (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    return best, result.image_shape


def search_xy_sweep(target: str, axis: str):
    """Sweep ±5cm in X or Y direction looking for object.
    
    Args:
        target: YOLO prompt
        axis: 'x' or 'y'
    
    Returns:
        (Detection, image_shape) or (None, None) if not found.
    """
    step = SEARCH_XY_STEP_M
    
    # Move +5cm
    if axis == 'x':
        arm.move_delta(dx=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    else:
        arm.move_delta(dy=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    time.sleep(0.2)
    
    det, shape = detect_object_2d(target)
    if det is not None:
        print(f"      Found at +{step*100:.0f}cm {axis.upper()}")
        # Move back to center
        if axis == 'x':
            arm.move_delta(dx=-step, frame="base", duration=SEARCH_WIGGLE_DURATION)
        else:
            arm.move_delta(dy=-step, frame="base", duration=SEARCH_WIGGLE_DURATION)
        return det, shape
    
    # Move -10cm (to -5cm from original)
    if axis == 'x':
        arm.move_delta(dx=-2*step, frame="base", duration=SEARCH_WIGGLE_DURATION * 1.5)
    else:
        arm.move_delta(dy=-2*step, frame="base", duration=SEARCH_WIGGLE_DURATION * 1.5)
    time.sleep(0.2)
    
    det, shape = detect_object_2d(target)
    if det is not None:
        print(f"      Found at -{step*100:.0f}cm {axis.upper()}")
        # Move back to center
        if axis == 'x':
            arm.move_delta(dx=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
        else:
            arm.move_delta(dy=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
        return det, shape
    
    # Move back to center
    if axis == 'x':
        arm.move_delta(dx=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    else:
        arm.move_delta(dy=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    
    return None, None


def search_at_current_rotation(target: str):
    """Search for object at current rotation by sweeping XY.
    
    Returns:
        (Detection, image_shape) or (None, None) if not found.
    """
    # First check current position
    det, shape = detect_object_2d(target)
    if det is not None:
        return det, shape
    
    # Sweep X
    print(f"      Sweeping X ±{SEARCH_XY_STEP_M*100:.0f}cm...")
    det, shape = search_xy_sweep(target, 'x')
    if det is not None:
        return det, shape
    
    # Sweep Y
    print(f"      Sweeping Y ±{SEARCH_XY_STEP_M*100:.0f}cm...")
    det, shape = search_xy_sweep(target, 'y')
    if det is not None:
        return det, shape
    
    return None, None


def search_wiggle(target: str):
    """Search for object by rotating last joint ±30° and sweeping XY at each angle.
    
    At each rotation angle (+30°, center, -30°), also sweeps ±5cm in X and Y.
    
    If detection fails at current position:
      1. At center: sweep XY
      2. Rotate +30°, sweep XY
      3. Rotate -60° (to -30°), sweep XY
      4. If detection found, descend one step
      5. Return to original position
    
    Returns:
        (Detection, image_shape, found_at_wiggle) or (None, None, False) if not found.
        found_at_wiggle: True if object was found during wiggle search.
    """
    wiggle_rad = math.radians(SEARCH_WIGGLE_ANGLE_DEG)
    
    # Try center position first with XY sweep
    print(f"    Wiggle search: checking center with XY sweep...")
    det, shape = search_at_current_rotation(target)
    if det is not None:
        print(f"    Found object at center")
        return det, shape, False
    
    # Try +30° with XY sweep
    print(f"    Wiggle search: rotating +{SEARCH_WIGGLE_ANGLE_DEG}°...")
    arm.move_delta(dyaw=wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
    time.sleep(0.2)
    
    det, shape = search_at_current_rotation(target)
    if det is not None:
        print(f"    Found object at +{SEARCH_WIGGLE_ANGLE_DEG}° position")
        # Descend a step while we can see it
        print(f"    Descending {DESCEND_STEP_M*1000:.0f}mm...")
        try:
            arm.move_delta(dz=-DESCEND_STEP_M, frame="base", duration=SERVO_MOVE_DURATION)
        except ArmError:
            pass  # Floor contact, that's fine
        # Return to original position
        arm.move_delta(dyaw=-wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
        time.sleep(0.2)
        return det, shape, True
    
    # Try -60° (to get to -30° from original) with XY sweep
    print(f"    Wiggle search: rotating -{SEARCH_WIGGLE_ANGLE_DEG * 2}° (to -{SEARCH_WIGGLE_ANGLE_DEG}°)...")
    arm.move_delta(dyaw=-2*wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION * 1.5)
    time.sleep(0.2)
    
    det, shape = search_at_current_rotation(target)
    if det is not None:
        print(f"    Found object at -{SEARCH_WIGGLE_ANGLE_DEG}° position")
        # Descend a step while we can see it
        print(f"    Descending {DESCEND_STEP_M*1000:.0f}mm...")
        try:
            arm.move_delta(dz=-DESCEND_STEP_M, frame="base", duration=SERVO_MOVE_DURATION)
        except ArmError:
            pass  # Floor contact, that's fine
        # Return to original position
        arm.move_delta(dyaw=wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
        time.sleep(0.2)
        return det, shape, True
    
    # Not found, return to original
    print(f"    Object not found in wiggle search, returning to center...")
    arm.move_delta(dyaw=wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
    time.sleep(0.2)
    
    return None, None, False


def get_object_pixel_center(detection):
    """Get the object center in pixel coordinates.

    Uses the mask centroid if a mask is available (more accurate for
    irregular shapes), otherwise falls back to bbox center.

    Returns:
        (u, v) pixel coordinates of the object center.
    """
    if detection.mask is not None:
        mask = detection.mask
        # Threshold the confidence mask
        binary = (mask > 0.5).astype(np.float32)
        total = binary.sum()
        if total > 0:
            # Compute centroid: weighted average of pixel coordinates
            ys, xs = np.where(binary > 0)
            u = float(xs.mean())
            v = float(ys.mean())
            return u, v
    # Fallback: bbox center
    x1, y1, x2, y2 = detection.bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_object_orientation(mask):
    """Get the major axis angle of the object from its mask.

    Uses the covariance matrix of mask pixel coordinates to find the
    principal axes. The major axis is the direction of greatest spread.

    Returns:
        (angle_image, elongation) or (None, None) if mask too small.
        angle_image: radians, measured CCW from +U axis, in [-pi/2, pi/2].
        elongation: ratio of major to minor eigenvalue (1.0 = circular).
    """
    binary = (mask > 0.5)
    ys, xs = np.where(binary)
    if len(xs) < 20:
        return None, None
    cov = np.cov(xs.astype(np.float64), ys.astype(np.float64))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_idx = np.argmax(eigenvalues)
    minor_idx = 1 - major_idx
    major_vec = eigenvectors[:, major_idx]
    angle_image = np.arctan2(major_vec[1], major_vec[0])
    # Normalize to [-pi/2, pi/2] (180-deg ambiguity for an ellipse)
    if angle_image > np.pi / 2:
        angle_image -= np.pi
    elif angle_image < -np.pi / 2:
        angle_image += np.pi
    elongation = eigenvalues[major_idx] / (eigenvalues[minor_idx] + 1e-6)
    return angle_image, elongation


def image_angle_to_ee_yaw(angle_image):
    """Convert mask major axis angle to desired EE yaw (perpendicular).

    The gripper fingers should close across the narrow part of the object,
    so the target yaw is 90 deg from the major axis. Result is normalized
    to [-pi/2, pi/2].

    Args:
        angle_image: Major axis angle in image frame, in [-pi/2, pi/2].

    Returns:
        Target EE yaw delta in radians, in [-pi/2, pi/2].
    """
    yaw = angle_image + np.pi / 2
    # Normalize to [-pi/2, pi/2]
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw < -np.pi / 2:
        yaw += np.pi
    return yaw


def get_servo_target_pixel(image_shape, use_ee_frame: bool):
    """Return (tx, ty) — the pixel target for servoing.
    
    - Base frame phase: target = image center (no offset)
    - EE frame phase: target = gripper offset (compensate for camera-gripper offset)
    """
    h, w = image_shape[0], image_shape[1]
    if use_ee_frame:
        # EE frame: use gripper offset
        return w / 2.0 + GRIPPER_U_OFFSET, h / 2.0 + GRIPPER_V_OFFSET
    else:
        # Base frame: use image center
        return w / 2.0, h / 2.0


def pixel_error_to_ee_delta(u_err, v_err):
    """Convert pixel error to EE delta in base frame.

    Uses the configured gain matrix. Clamps the step size.

    Returns:
        (dx, dy, dz) in meters.
    """
    dx = GAIN_U_TO_DX * u_err + GAIN_V_TO_DX * v_err
    dy = GAIN_U_TO_DY * u_err
    dz = GAIN_V_TO_DZ * v_err

    # Clamp lateral step magnitude
    step_norm = np.sqrt(dx**2 + dy**2 + dz**2)
    if step_norm > MAX_LATERAL_STEP_M:
        scale = MAX_LATERAL_STEP_M / step_norm
        dx *= scale
        dy *= scale
        dz *= scale
    return dx, dy, dz


def clamp(value, low, high):
    return max(low, min(high, value))


# ============================================================================
# Auto-calibration (optional — run once to determine gain signs)
# ============================================================================

def auto_calibrate(target: str = TARGET_OBJECT, test_delta: float = 0.02):
    """Determine camera-to-EE mapping by making small test movements.

    Makes a small +Y move and a small +X move, observing how the object
    pixel center shifts. Prints the recommended gain signs.

    Args:
        target: Object to track during calibration
        test_delta: Size of test movement in meters (default: 2cm)
    """
    print("=== Auto-calibration ===")
    print(f"Tracking '{target}' during test moves of {test_delta*100:.0f}cm")

    det0, shape0 = detect_object_2d(target)
    if det0 is None:
        print("ERROR: Cannot detect object for calibration. Aborting.")
        return
    u0, v0 = get_object_pixel_center(det0)
    print(f"Baseline pixel: ({u0:.0f}, {v0:.0f})")

    # Test +Y movement (right in base frame)
    print(f"\nMoving EE +Y by {test_delta}m...")
    arm.move_delta(dy=test_delta, frame="base", duration=SERVO_MOVE_DURATION)
    time.sleep(0.3)

    det_y, _ = detect_object_2d(target)
    if det_y:
        uy, vy = get_object_pixel_center(det_y)
        du_dy = (uy - u0) / test_delta
        dv_dy = (vy - v0) / test_delta
        print(f"After +Y: pixel ({uy:.0f}, {vy:.0f})")
        print(f"  du/dy = {du_dy:.0f} px/m,  dv/dy = {dv_dy:.0f} px/m")
    else:
        print("WARNING: Lost object after +Y move")
        du_dy, dv_dy = 0, 0

    arm.move_delta(dy=-test_delta, frame="base", duration=SERVO_MOVE_DURATION)
    time.sleep(0.3)

    # Test +X movement (forward in base frame)
    print(f"\nMoving EE +X by {test_delta}m...")
    arm.move_delta(dx=test_delta, frame="base", duration=SERVO_MOVE_DURATION)
    time.sleep(0.3)

    det_x, _ = detect_object_2d(target)
    if det_x:
        ux, vx = get_object_pixel_center(det_x)
        du_dx = (ux - u0) / test_delta
        dv_dx = (vx - v0) / test_delta
        print(f"After +X: pixel ({ux:.0f}, {vx:.0f})")
        print(f"  du/dx = {du_dx:.0f} px/m,  dv/dx = {dv_dx:.0f} px/m")
    else:
        print("WARNING: Lost object after +X move")
        du_dx, dv_dx = 0, 0

    arm.move_delta(dx=-test_delta, frame="base", duration=SERVO_MOVE_DURATION)
    time.sleep(0.3)

    print("\n=== Recommended gains ===")
    if abs(du_dy) > 10:
        print(f"GAIN_U_TO_DY = {1.0 / du_dy:.6f}  (u_err -> dy)")
    else:
        print("GAIN_U_TO_DY: inconclusive")
    if abs(dv_dx) > 10:
        print(f"GAIN_V_TO_DX = {1.0 / dv_dx:.6f}  (v_err -> dx)")
    else:
        print("GAIN_V_TO_DX: inconclusive")
    if abs(du_dx) > 10:
        print(f"GAIN_U_TO_DX = {1.0 / du_dx:.6f}  (u_err -> dx, cross-term)")
    if abs(dv_dy) > 10:
        print(f"GAIN_V_TO_DZ = {1.0 / dv_dy:.6f}  (v_err -> dz, cross-term)")
    print("\nCalibration complete. Update the gains at the top of the script.")


# ============================================================================
# Main pick pipeline
# ============================================================================

def servo_descend(target: str = TARGET_OBJECT):
    """Servo-descend loop: correct lateral error AND descend simultaneously.

    Uses 2D detection only (no depth). Descends until ArmError (convergence
    timeout = floor contact).

    Two phases:
      - Above EE_FRAME_Z_THRESHOLD: base frame servoing, target = image center
      - Below EE_FRAME_Z_THRESHOLD: EE frame servoing, target = gripper offset

    On detection miss:
      - Wiggle last joint ±30° AND sweep XY ±5cm to search for object
      - If found during wiggle, descend a step and continue
      - Only abort after MAX_SEARCH_FAILURES consecutive failed searches

    Returns:
        True if floor contact detected. False if lost object after all retries.
    """
    ee_x, ee_y, ee_z = sensors.get_ee_position()
    accumulated_yaw = 0.0  # Track total yaw applied for logging
    consecutive_search_failures = 0

    print(f"\n--- Servo-Descend: approaching '{target}' ---")
    print(f"  Current EE Z: {ee_z:.3f}m, descend step: {DESCEND_STEP_M*1000:.0f}mm")
    print(f"  EE frame switch at Z < {EE_FRAME_Z_THRESHOLD}m")
    print(f"  Search: ±{SEARCH_WIGGLE_ANGLE_DEG}° rotation + ±{SEARCH_XY_STEP_M*100:.0f}cm XY sweep")
    print(f"  Max search failures: {MAX_SEARCH_FAILURES}")
    display.show_text(f"Approaching {target}...")
    display.show_face("thinking")

    for i in range(MAX_SERVO_ITERATIONS):
        ee_x, ee_y, ee_z = sensors.get_ee_position()
        use_ee_frame = ee_z < EE_FRAME_Z_THRESHOLD

        # Detect object (2D with mask)
        det, shape = detect_object_2d(target)

        if det is None:
            print(f"  Iter {i+1}: object not detected, initiating search wiggle...")
            det, shape, found_at_wiggle = search_wiggle(target)
            
            if det is None:
                consecutive_search_failures += 1
                print(f"    Search failed ({consecutive_search_failures}/{MAX_SEARCH_FAILURES})")
                if consecutive_search_failures >= MAX_SEARCH_FAILURES:
                    print("  ERROR: Lost object after max search attempts. Aborting.")
                    return False
                # Continue loop, try again next iteration
                continue
            else:
                # Found during wiggle - reset failure counter
                consecutive_search_failures = 0
                if found_at_wiggle:
                    # Already descended during wiggle, skip to next iteration
                    print(f"    Continuing after wiggle descent...")
                    continue
        else:
            consecutive_search_failures = 0

        # Compute pixel error using mask centroid
        # Target depends on phase: image center (base) or gripper offset (EE)
        obj_u, obj_v = get_object_pixel_center(det)
        cx, cy = get_servo_target_pixel(shape, use_ee_frame)
        u_err = obj_u - cx
        v_err = obj_v - cy
        error_mag = np.sqrt(u_err**2 + v_err**2)

        # Compute orientation if mask available (only in EE frame)
        dyaw = 0.0
        orientation_str = ""
        if use_ee_frame and det.mask is not None:
            angle_img, elongation = get_object_orientation(det.mask)
            if angle_img is not None and elongation > EE_MIN_ELONGATION:
                target_yaw = image_angle_to_ee_yaw(angle_img)
                dyaw = EE_YAW_GAIN * target_yaw
                orientation_str = (f", angle={np.degrees(angle_img):.0f}deg"
                                   f" elong={elongation:.1f}"
                                   f" dyaw={np.degrees(dyaw):.1f}deg")

        has_mask = det.mask is not None
        src = "mask" if has_mask else "bbox"
        frame_str = "EE" if use_ee_frame else "BASE"
        target_str = "gripper" if use_ee_frame else "center"
        print(f"  Iter {i+1}: pixel err=({u_err:.0f},{v_err:.0f}) "
              f"|{error_mag:.0f}px| [{src}] [{frame_str}→{target_str}], "
              f"EE Z={ee_z:.3f}m{orientation_str}")

        # Compute lateral correction based on current frame
        if use_ee_frame:
            # EE frame gains from calibration
            dx_lat = EE_GAIN_V_TO_DX * v_err
            dy_lat = EE_GAIN_U_TO_DY * u_err
            # Clamp
            lat_norm = np.sqrt(dx_lat**2 + dy_lat**2)
            if lat_norm > MAX_LATERAL_STEP_M:
                scale = MAX_LATERAL_STEP_M / lat_norm
                dx_lat *= scale
                dy_lat *= scale
        else:
            # Base frame gains
            dx_lat, dy_lat, _ = pixel_error_to_ee_delta(u_err, v_err)

        # Decide whether to also descend this step
        descend_this_step = 0.0
        if error_mag < DESCEND_PAUSE_PIXELS:
            descend_this_step = DESCEND_STEP_M
        else:
            print(f"    Pausing descent (error {error_mag:.0f} > "
                  f"{DESCEND_PAUSE_PIXELS}px), centering first...")

        # Combine lateral + descent into one move
        dx = dx_lat
        dy = dy_lat
        # In base frame: -Z = descend. In EE frame: +Z = descend (EE Z points toward floor).
        dz = descend_this_step if use_ee_frame else -descend_this_step

        # If move is negligible, force a descent step
        total_step = np.sqrt(dx**2 + dy**2 + dz**2)
        if total_step < MIN_LATERAL_STEP_M:
            dz = DESCEND_STEP_M if use_ee_frame else -DESCEND_STEP_M
            descend_this_step = DESCEND_STEP_M
            dx, dy = 0.0, 0.0

        desc_str = f", descend={descend_this_step*1000:.0f}mm" if descend_this_step > 0 else ""
        yaw_str = f", dyaw={np.degrees(dyaw):.1f}deg" if dyaw != 0 else ""
        print(f"    Move [{frame_str}]: dx={dx*1000:.1f}mm, dy={dy*1000:.1f}mm, "
              f"dz={dz*1000:.1f}mm{desc_str}{yaw_str}")

        frame = "ee" if use_ee_frame else "base"
        try:
            arm.move_delta(dx=dx, dy=dy, dz=dz, droll=0, dpitch=0, dyaw=dyaw,
                           frame=frame, duration=SERVO_MOVE_DURATION)
        except ArmError as e:
            print(f"  FLOOR CONTACT: arm could not reach target ({e})")
            print(f"  Final EE Z: {sensors.get_ee_position()[2]:.3f}m")
            if dyaw != 0:
                accumulated_yaw += dyaw
                print(f"  Total yaw applied: {np.degrees(accumulated_yaw):.1f}deg")
            return True

        if dyaw != 0:
            accumulated_yaw += dyaw

        time.sleep(0.2)

    ee_x, ee_y, ee_z = sensors.get_ee_position()
    print(f"  WARNING: Max iterations ({MAX_SERVO_ITERATIONS}) reached "
          f"(EE Z={ee_z:.3f}m).")
    return True


def pick_up_object(target: str = TARGET_OBJECT):
    """Full pick pipeline: detect → servo-descend → grasp → lift.

    The servo-descend phase simultaneously corrects lateral centering error
    and descends toward the object, keeping it centered in the camera image
    throughout the entire approach. Uses 2D YOLO only with mask centroid.

    Args:
        target: YOLO text prompt for the object to pick
    """
    print(f"=== Pick Object: '{target}' ===\n")

    # --- Phase 0: Initialize gripper ---
    print("Phase 0: Initializing gripper...")
    display.show_text(f"Picking up {target}")
    display.show_face("thinking")
    gripper.activate()
    gripper.open()
    time.sleep(0.5)

    # Tilt EE +20 deg around Y axis so camera points more downward
    print("Tilting EE -20 deg pitch (camera down)...")
    arm.move_delta(dpitch=math.radians(-20), frame="ee", duration=1.0)
    time.sleep(0.3)

    # --- Phase 1: Initial detection with search wiggle ---
    print("\nPhase 1: Initial detection...")
    det, shape = detect_object_2d(target)

    if det is None:
        print("  Object not detected at center, trying search wiggle...")
        det, shape, _ = search_wiggle(target)

    if det is None:
        print("ERROR: Object not detected after search. Aborting.")
        display.show_text(f"{target} not found!")
        display.show_face("sad")
        return False

    obj_u, obj_v = get_object_pixel_center(det)
    has_mask = det.mask is not None
    src = "mask centroid" if has_mask else "bbox center"
    print(f"  Detected '{det.class_name}' at pixel ({obj_u:.0f}, {obj_v:.0f}) [{src}]")

    # --- Phase 2: Servo-descend (center + descend simultaneously) ---
    print("\nPhase 2: Servo-descend (centering + descending together)...")
    display.show_text(f"Approaching {target}...")
    reached = servo_descend(target)

    if not reached:
        print("WARNING: Servo-descend did not fully converge, attempting grasp anyway.")

    # --- Phase 3: Grasp ---
    print("\nPhase 3: Grasping...")
    display.show_text(f"Grasping {target}...")

    grasped = gripper.grasp(speed=GRASP_SPEED, force=GRASP_FORCE)
    time.sleep(0.5)

    if grasped:
        print("  Object grasped!")
        display.show_face("happy")
        display.show_text(f"Got the {target}!")
    else:
        print("  WARNING: No object detected in gripper.")
        display.show_face("concerned")
        display.show_text("Grasp missed...")

    # --- Phase 4: Go home ---
    print("\nPhase 4: Going home...")
    arm.go_home()
    time.sleep(0.5)

    # --- Phase 5: Release ---
    print("\nPhase 5: Opening gripper...")
    gripper.open()
    time.sleep(0.5)

    if grasped:
        print(f"\n=== Successfully picked '{target}' and returned home! ===")
        display.show_face("excited")
    else:
        print(f"\n=== Pick attempt for '{target}' complete (grasp uncertain). ===")

    return grasped


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__" or True:  # Always run when exec'd via code execution
    # To calibrate first (uncomment):
    # auto_calibrate()

    success = pick_up_object(TARGET_OBJECT)
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
