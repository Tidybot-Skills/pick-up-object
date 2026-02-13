"""Pick object with image-based visual servoing (IBVS) — 2D only.

Continuously centers the target object in the camera image while descending
toward it, then grasps it. Uses YOLO 2D segmentation with mask centroid
for sub-bbox-level accuracy.

Workflow:
  1. Detect target with YOLO 2D (with mask) → get pixel center
  2. Servo-descend loop: at every step, detect → correct lateral error AND
     descend a small increment. Keeps object centered throughout approach.
  3. On detection miss: sweep XY ±5cm and rotate ±30° to search
     - If found, STAY at that position and continue descent
  4. Grasp after descending to floor contact
  5. Lift and return home

Usage (submit via code execution API):
  curl -X POST localhost:8080/code/execute \
    -H "X-Lease-Id: <lease>" \
    -H "Content-Type: application/json" \
    -d '{"code": "exec(open(\"pick_up_object.py\").read())"}'
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
GAIN_U_TO_DY = -0.0006  # u_err → dy
GAIN_U_TO_DX = 0.0
GAIN_V_TO_DX = -0.0006  # v_err → dx
GAIN_V_TO_DZ = 0.0

# --- Gripper-to-camera offset (EE frame only) ---
GRIPPER_U_OFFSET = 0.0
GRIPPER_V_OFFSET = -120  # pixels

# --- Gradual gripper offset transition ---
OFFSET_START_Z = 0.0    # Start applying gripper offset
OFFSET_END_Z = -0.25    # Full offset applied

# --- Servoing loop parameters ---
PIXEL_TOLERANCE = 30
MAX_SERVO_ITERATIONS = 200
MAX_LATERAL_STEP_M = 0.05
MIN_LATERAL_STEP_M = 0.001
SERVO_MOVE_DURATION = 0.5

# --- Search parameters ---
SEARCH_WIGGLE_ANGLE_DEG = 30
SEARCH_XY_STEP_M = 0.05
SEARCH_WIGGLE_DURATION = 0.4
MAX_SEARCH_FAILURES = 3

# --- Descent parameters ---
DESCEND_STEP_M = 0.05
DESCEND_PAUSE_PIXELS = 80

# --- EE frame mode ---
EE_FRAME_Z_THRESHOLD = -0.25  # Same as OFFSET_END_Z
EE_GAIN_U_TO_DY = +1.0 / 580
EE_GAIN_V_TO_DX = -1.0 / 660

# --- Straight-down orientation (roll, pitch, yaw in radians) ---
# For Franka Panda, pointing EE straight down
STRAIGHT_DOWN_ROLL = math.pi
STRAIGHT_DOWN_PITCH = 0.0
STRAIGHT_DOWN_YAW = 0.0

# --- Grasp parameters ---
GRASP_FORCE = 50
GRASP_SPEED = 200

# --- Joint 7 rotation control (EE frame) ---
J7_ROTATION_GAIN = 0.3        # Fraction of PCA error to correct per step (reduced for stability)
J7_MIN_CORRECTION_RAD = 0.03  # ~2° deadband
J7_MAX_CORRECTION_RAD = 0.15  # ~8.5° max per step (reduced from 17°)


# ============================================================================
# Helper functions
# ============================================================================

def detect_object_2d(target: str, confidence: float = DETECTION_CONFIDENCE):
    """Detect target object with mask and return the best detection."""
    result = yolo.segment_camera(
        target, camera_id=CAMERA_ID, confidence=confidence,
        save_visualization=True, mask_format="npz",
    )
    detections = result.get_by_class(target)
    if not detections:
        detections = result.detections
    if not detections:
        return None, None
    best = max(detections, key=lambda d: d.area if d.area > 0 else
               (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    return best, result.image_shape


def point_straight_down():
    """Set EE orientation to point straight down (no rotation)."""
    print("    Setting EE orientation straight down...")
    arm.move_to_pose(
        roll=STRAIGHT_DOWN_ROLL,
        pitch=STRAIGHT_DOWN_PITCH,
        yaw=STRAIGHT_DOWN_YAW,
        duration=1.0
    )
    time.sleep(0.3)


def search_xy_sweep(target: str, axis: str):
    """Sweep ±5cm in X or Y direction looking for object.
    
    If found, STAYS at that position (does not return to center).
    
    Returns:
        (Detection, image_shape, offset_m) or (None, None, 0) if not found.
        offset_m: The offset where object was found (positive or negative).
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
        print(f"      Found at +{step*100:.0f}cm {axis.upper()} - staying here")
        return det, shape, step
    
    # Move -10cm (to -5cm from original)
    if axis == 'x':
        arm.move_delta(dx=-2*step, frame="base", duration=SEARCH_WIGGLE_DURATION * 1.5)
    else:
        arm.move_delta(dy=-2*step, frame="base", duration=SEARCH_WIGGLE_DURATION * 1.5)
    time.sleep(0.2)
    
    det, shape = detect_object_2d(target)
    if det is not None:
        print(f"      Found at -{step*100:.0f}cm {axis.upper()} - staying here")
        return det, shape, -step
    
    # Not found - move back to center
    if axis == 'x':
        arm.move_delta(dx=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    else:
        arm.move_delta(dy=step, frame="base", duration=SEARCH_WIGGLE_DURATION)
    
    return None, None, 0


def search_at_current_rotation(target: str):
    """Search for object at current rotation by sweeping XY.
    
    If found at an offset, STAYS at that position.
    
    Returns:
        (Detection, image_shape) or (None, None) if not found.
    """
    # First check current position
    det, shape = detect_object_2d(target)
    if det is not None:
        return det, shape
    
    # Sweep X
    print(f"      Sweeping X ±{SEARCH_XY_STEP_M*100:.0f}cm...")
    det, shape, _ = search_xy_sweep(target, 'x')
    if det is not None:
        return det, shape
    
    # Sweep Y
    print(f"      Sweeping Y ±{SEARCH_XY_STEP_M*100:.0f}cm...")
    det, shape, _ = search_xy_sweep(target, 'y')
    if det is not None:
        return det, shape
    
    return None, None


def search_wiggle(target: str):
    """Search for object by rotating ±30° and sweeping XY at each angle.
    
    If found, STAYS at that position and descends one step.
    Rotation is returned to center (0°) but XY offset is kept.
    
    Returns:
        (Detection, image_shape, found_at_wiggle) or (None, None, False) if not found.
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
        print(f"    Found at +{SEARCH_WIGGLE_ANGLE_DEG}° - descending & returning rotation to 0°")
        try:
            arm.move_delta(dz=-DESCEND_STEP_M, frame="base", duration=SERVO_MOVE_DURATION)
        except ArmError:
            pass
        # Return rotation to center (but keep XY position)
        arm.move_delta(dyaw=-wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
        time.sleep(0.2)
        return det, shape, True
    
    # Try -60° (to get to -30°) with XY sweep
    print(f"    Wiggle search: rotating to -{SEARCH_WIGGLE_ANGLE_DEG}°...")
    arm.move_delta(dyaw=-2*wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION * 1.5)
    time.sleep(0.2)
    
    det, shape = search_at_current_rotation(target)
    if det is not None:
        print(f"    Found at -{SEARCH_WIGGLE_ANGLE_DEG}° - descending & returning rotation to 0°")
        try:
            arm.move_delta(dz=-DESCEND_STEP_M, frame="base", duration=SERVO_MOVE_DURATION)
        except ArmError:
            pass
        # Return rotation to center
        arm.move_delta(dyaw=wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
        time.sleep(0.2)
        return det, shape, True
    
    # Not found, return rotation to center
    print(f"    Object not found, returning rotation to center...")
    arm.move_delta(dyaw=wiggle_rad, frame="ee", duration=SEARCH_WIGGLE_DURATION)
    time.sleep(0.2)
    
    return None, None, False


def get_object_pixel_center(detection):
    """Get object center in pixel coordinates (mask centroid or bbox center)."""
    if detection.mask is not None:
        mask = detection.mask
        binary = (mask > 0.5).astype(np.float32)
        total = binary.sum()
        if total > 0:
            ys, xs = np.where(binary > 0)
            return float(xs.mean()), float(ys.mean())
    x1, y1, x2, y2 = detection.bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_mask_orientation(detection):
    """Get object orientation angle from mask using PCA.
    
    Returns angle in radians for EE yaw rotation to align gripper PERPENDICULAR 
    to object's long axis (for grasping across the object).
    Clamped to ±90° to avoid joint limits.
    Returns 0 if no mask or can't compute.
    """
    if detection.mask is None:
        return 0.0
    
    mask = detection.mask
    binary = (mask > 0.5).astype(np.float32)
    ys, xs = np.where(binary > 0)
    
    if len(xs) < 10:  # Need enough pixels
        return 0.0
    
    # Center the points
    cx, cy = xs.mean(), ys.mean()
    xs_c = xs - cx
    ys_c = ys - cy
    
    # Compute covariance matrix
    cov_xx = np.mean(xs_c * xs_c)
    cov_yy = np.mean(ys_c * ys_c)
    cov_xy = np.mean(xs_c * ys_c)
    
    # Principal axis angle (eigenvector of largest eigenvalue)
    theta = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)
    
    # Add 90° to get perpendicular (gripper grabs across, not along)
    theta_perp = theta + math.pi / 2
    
    # Normalize to [-pi, pi]
    while theta_perp > math.pi:
        theta_perp -= 2 * math.pi
    while theta_perp < -math.pi:
        theta_perp += 2 * math.pi
    
    # Clamp to ±90° to avoid joint limits
    max_angle = math.pi / 2
    theta_perp = max(-max_angle, min(max_angle, theta_perp))
    
    # Negate for image-to-EE coordinate conversion
    return -theta_perp


def get_pca_angle_raw(detection):
    """Get RAW principal axis angle from mask (no perpendicular, no clamp).
    
    Returns angle in radians in image frame. Used for continuous tracking.
    The gripper should align perpendicular to this angle.
    Returns None if no mask or can't compute.
    """
    if detection.mask is None:
        return None
    
    mask = detection.mask
    binary = (mask > 0.5).astype(np.float32)
    ys, xs = np.where(binary > 0)
    
    if len(xs) < 10:
        return None
    
    cx, cy = xs.mean(), ys.mean()
    xs_c = xs - cx
    ys_c = ys - cy
    
    cov_xx = np.mean(xs_c * xs_c)
    cov_yy = np.mean(ys_c * ys_c)
    cov_xy = np.mean(xs_c * ys_c)
    
    # Principal axis angle in image frame
    theta = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)
    return theta


def compute_j7_correction(detection, current_j7: float) -> tuple:
    """Compute joint 7 correction to align gripper perpendicular to object.
    
    Returns: (correction_rad, debug_info) where debug_info is a string.
    """
    pca_angle = get_pca_angle_raw(detection)
    if pca_angle is None:
        return 0.0, "no_mask"
    
    pca_deg = math.degrees(pca_angle)
    j7_deg = math.degrees(current_j7)
    
    # Target: J7 should match PCA angle (so gripper is perpendicular to object)
    # When J7 = PCA, the gripper crosses the object's long axis at 90°
    # Note: This is a simplification; exact mapping depends on arm pose
    
    # Simple approach: drive J7 toward PCA angle
    # (The perpendicular is handled by the coordinate frame convention)
    error = pca_angle - current_j7
    
    # Normalize error to [-pi, pi]
    while error > math.pi:
        error -= 2 * math.pi
    while error < -math.pi:
        error += 2 * math.pi
    
    error_deg = math.degrees(error)
    
    # Apply gain
    correction = error * J7_ROTATION_GAIN
    
    # Deadband
    if abs(correction) < J7_MIN_CORRECTION_RAD:
        return 0.0, f"pca={pca_deg:.0f}° j7={j7_deg:.0f}° err={error_deg:.0f}° (deadband)"
    
    # Clamp max correction per step
    correction = max(-J7_MAX_CORRECTION_RAD, min(J7_MAX_CORRECTION_RAD, correction))
    
    corr_deg = math.degrees(correction)
    return correction, f"pca={pca_deg:.0f}° j7={j7_deg:.0f}° err={error_deg:.0f}° corr={corr_deg:+.0f}°"


def get_servo_target_pixel(image_shape, ee_z: float):
    """Return servo target pixel with gradual gripper offset.
    
    Offset interpolates linearly from 0% at OFFSET_START_Z to 100% at OFFSET_END_Z.
    This prevents abrupt target jumps when transitioning to EE frame.
    """
    h, w = image_shape[0], image_shape[1]
    
    # Compute offset ratio (0 to 1) based on Z height
    if ee_z >= OFFSET_START_Z:
        ratio = 0.0
    elif ee_z <= OFFSET_END_Z:
        ratio = 1.0
    else:
        ratio = (OFFSET_START_Z - ee_z) / (OFFSET_START_Z - OFFSET_END_Z)
    
    u_offset = GRIPPER_U_OFFSET * ratio
    v_offset = GRIPPER_V_OFFSET * ratio
    
    return w / 2.0 + u_offset, h / 2.0 + v_offset


def pixel_error_to_ee_delta(u_err, v_err):
    """Convert pixel error to EE delta in base frame."""
    dx = GAIN_U_TO_DX * u_err + GAIN_V_TO_DX * v_err
    dy = GAIN_U_TO_DY * u_err
    dz = GAIN_V_TO_DZ * v_err
    
    step_norm = np.sqrt(dx**2 + dy**2 + dz**2)
    if step_norm > MAX_LATERAL_STEP_M:
        scale = MAX_LATERAL_STEP_M / step_norm
        dx *= scale
        dy *= scale
        dz *= scale
    return dx, dy, dz


# ============================================================================
# Main pick pipeline
# ============================================================================

def servo_descend(target: str = TARGET_OBJECT):
    """Servo-descend loop with two-phase control.
    
    BASE frame (Z > threshold):
      - Target = image center
      - Wiggle search on detection miss
      
    EE frame (Z <= threshold):
      - Target = gripper offset
      - Continuous joint 7 rotation to track object orientation
      - NO wiggle search (abort on detection miss)
    """
    ee_x, ee_y, ee_z = sensors.get_ee_position()
    consecutive_search_failures = 0

    print(f"\n--- Servo-Descend: approaching '{target}' ---")
    print(f"  Current EE Z: {ee_z:.3f}m, descend step: {DESCEND_STEP_M*1000:.0f}mm")
    print(f"  EE frame switch at Z < {EE_FRAME_Z_THRESHOLD}m")
    print(f"  BASE frame: wiggle search enabled")
    print(f"  EE frame: continuous J7 rotation tracking, no wiggle")
    display.show_text(f"Approaching {target}...")
    display.show_face("thinking")

    for i in range(MAX_SERVO_ITERATIONS):
        ee_x, ee_y, ee_z = sensors.get_ee_position()
        use_ee_frame = ee_z < EE_FRAME_Z_THRESHOLD
        
        # Get current joint 7 for rotation tracking
        joints = sensors.get_arm_joints()
        current_j7 = joints[6]

        # Detect object
        det, shape = detect_object_2d(target)

        if det is None:
            if use_ee_frame:
                # EE frame: no wiggle, just fail after a few misses
                consecutive_search_failures += 1
                print(f"  Iter {i+1}: object not detected in EE frame ({consecutive_search_failures}/{MAX_SEARCH_FAILURES})")
                if consecutive_search_failures >= MAX_SEARCH_FAILURES:
                    print("  ERROR: Lost object in EE frame. Aborting.")
                    return False
                time.sleep(0.2)
                continue
            else:
                # BASE frame: use wiggle search
                print(f"  Iter {i+1}: object not detected, searching...")
                det, shape, found_at_wiggle = search_wiggle(target)
                
                if det is None:
                    consecutive_search_failures += 1
                    print(f"    Search failed ({consecutive_search_failures}/{MAX_SEARCH_FAILURES})")
                    if consecutive_search_failures >= MAX_SEARCH_FAILURES:
                        print("  ERROR: Lost object after max search attempts.")
                        return False
                    continue
                else:
                    consecutive_search_failures = 0
                    if found_at_wiggle:
                        print(f"    Continuing after search descent...")
                        continue
        else:
            consecutive_search_failures = 0

        # Compute pixel error
        obj_u, obj_v = get_object_pixel_center(det)
        cx, cy = get_servo_target_pixel(shape, ee_z)
        u_err = obj_u - cx
        v_err = obj_v - cy
        error_mag = np.sqrt(u_err**2 + v_err**2)

        has_mask = det.mask is not None
        src = "mask" if has_mask else "bbox"
        frame_str = "EE" if use_ee_frame else "BASE"
        target_str = "gripper" if use_ee_frame else "center"
        
        # Compute J7 rotation correction (EE frame only)
        j7_correction = 0.0
        j7_debug = ""
        if use_ee_frame and has_mask:
            j7_correction, j7_debug = compute_j7_correction(det, current_j7)
        
        j7_str = f" [{j7_debug}]" if j7_debug else ""
        print(f"  Iter {i+1}: err=({u_err:.0f},{v_err:.0f}) |{error_mag:.0f}px| "
              f"[{src}] [{frame_str}→{target_str}] Z={ee_z:.3f}m{j7_str}")

        # Compute lateral correction
        if use_ee_frame:
            dx_lat = EE_GAIN_V_TO_DX * v_err
            dy_lat = EE_GAIN_U_TO_DY * u_err
            lat_norm = np.sqrt(dx_lat**2 + dy_lat**2)
            if lat_norm > MAX_LATERAL_STEP_M:
                scale = MAX_LATERAL_STEP_M / lat_norm
                dx_lat *= scale
                dy_lat *= scale
        else:
            dx_lat, dy_lat, _ = pixel_error_to_ee_delta(u_err, v_err)

        # Descend if centered enough
        descend_this_step = 0.0
        if error_mag < DESCEND_PAUSE_PIXELS:
            descend_this_step = DESCEND_STEP_M
        else:
            print(f"    Pausing descent (error {error_mag:.0f} > {DESCEND_PAUSE_PIXELS}px)")

        dx = dx_lat
        dy = dy_lat
        dz = descend_this_step if use_ee_frame else -descend_this_step

        # Force descent if move is negligible
        if np.sqrt(dx**2 + dy**2 + dz**2) < MIN_LATERAL_STEP_M and j7_correction == 0:
            dz = DESCEND_STEP_M if use_ee_frame else -DESCEND_STEP_M
            descend_this_step = DESCEND_STEP_M
            dx, dy = 0.0, 0.0

        desc_str = f" ↓{descend_this_step*1000:.0f}mm" if descend_this_step > 0 else ""
        print(f"    Move [{frame_str}]: dx={dx*1000:.1f} dy={dy*1000:.1f} dz={dz*1000:.1f}{desc_str}")

        # Execute moves
        frame = "ee" if use_ee_frame else "base"
        
        # In EE frame: do J7 correction first, then XY+Z
        if use_ee_frame and j7_correction != 0:
            new_j7 = current_j7 + j7_correction
            new_joints = list(joints)
            new_joints[6] = new_j7
            print(f"    J7: {math.degrees(current_j7):.1f}° → {math.degrees(new_j7):.1f}°")
            try:
                arm.move_joints(new_joints, duration=0.3)
            except ArmError as e:
                print(f"    J7 move failed: {e}")
            time.sleep(0.1)
        
        # XY + Z move
        try:
            arm.move_delta(dx=dx, dy=dy, dz=dz, frame=frame, duration=SERVO_MOVE_DURATION)
        except ArmError as e:
            print(f"  FLOOR CONTACT: {e}")
            print(f"  Final EE Z: {sensors.get_ee_position()[2]:.3f}m")
            return True

        time.sleep(0.2)

    print(f"  WARNING: Max iterations ({MAX_SERVO_ITERATIONS}) reached")
    return True


def pick_up_object(target: str = TARGET_OBJECT):
    """Full pick pipeline: detect → servo-descend → grasp → lift."""
    print(f"=== Pick Object: '{target}' ===\n")

    # Phase 0: Initialize gripper
    print("Phase 0: Initializing gripper...")
    display.show_text(f"Picking up {target}")
    display.show_face("thinking")
    gripper.activate()
    gripper.open()
    time.sleep(0.5)

    # Tilt camera down
    print("Tilting EE -20 deg pitch (camera down)...")
    arm.move_delta(dpitch=math.radians(-20), frame="ee", duration=1.0)
    time.sleep(0.3)

    # Phase 1: Initial detection
    print("\nPhase 1: Initial detection...")
    det, shape = detect_object_2d(target)

    if det is None:
        print("  Object not detected, trying search...")
        det, shape, _ = search_wiggle(target)

    if det is None:
        print("ERROR: Object not detected after search. Aborting.")
        display.show_text(f"{target} not found!")
        display.show_face("sad")
        return False

    obj_u, obj_v = get_object_pixel_center(det)
    src = "mask" if det.mask is not None else "bbox"
    print(f"  Detected at pixel ({obj_u:.0f}, {obj_v:.0f}) [{src}]")

    # Phase 2: Servo-descend
    print("\nPhase 2: Servo-descend...")
    display.show_text(f"Approaching {target}...")
    reached = servo_descend(target)

    if not reached:
        print("WARNING: Servo-descend incomplete, attempting grasp anyway.")

    # Phase 3: Grasp
    print("\nPhase 3: Grasping...")
    display.show_text(f"Grasping {target}...")
    grasped = gripper.grasp(speed=GRASP_SPEED, force=GRASP_FORCE)
    time.sleep(0.5)

    if grasped:
        print("  Object grasped!")
        display.show_face("happy")
    else:
        print("  WARNING: No object detected in gripper.")
        display.show_face("concerned")

    # Phase 4: Go home
    print("\nPhase 4: Going home...")
    arm.go_home()
    time.sleep(0.5)

    # Phase 5: Release
    print("\nPhase 5: Opening gripper...")
    gripper.open()
    time.sleep(0.5)

    if grasped:
        print(f"\n=== Successfully picked '{target}'! ===")
        display.show_face("excited")
    else:
        print(f"\n=== Pick attempt complete (grasp uncertain). ===")

    return grasped


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__" or True:
    success = pick_up_object(TARGET_OBJECT)
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
