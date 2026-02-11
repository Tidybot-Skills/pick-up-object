"""
pick-up-object - IBVS visual servoing to pick up objects from the ground

Uses wrist camera + YOLO detection for bbox-based centering,
then two-stage descent to ground level for grasp.

Tested with banana on ground (z=-0.65 in arm base frame).
"""
import time
from robot_sdk import yolo, arm, gripper, sensors

IMG_CX, IMG_CY = 320, 240
GROUND_Z = -0.65  # Ground level in arm base frame

def get_gains(bbox):
    """Adaptive gains based on distance (bbox size)"""
    if bbox < 80:
        return 0.00025, 0.020  # Far: faster gains
    elif bbox < 120:
        return 0.00018, 0.012  # Mid: moderate
    else:
        return 0.00012, 0.006  # Close: fine control

CENTER_THRESH = 50
BBOX_READY = 130
MAX_ITERS = 100

def detect(target="banana"):
    """Detect target object, return (cx, cy, bbox_size) or None"""
    r = yolo.segment_camera(target, camera_id="wrist_cam", confidence=0.15)
    if not r.detections:
        return None
    b = r.detections[0].bbox
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    size = max(b[2] - b[0], b[3] - b[1])
    return (cx, cy, size)

def run_ibvs(target="banana"):
    """Visual servo to center target in camera frame"""
    print(f"=== IBVS: centering {target} ===")
    
    try:
        gripper.activate()
    except:
        pass
    gripper.open()
    time.sleep(0.3)
    
    ready_count = 0
    lost_count = 0
    
    for i in range(MAX_ITERS):
        d = detect(target)
        
        if d is None:
            lost_count += 1
            if lost_count > 10:
                print("Lost target")
                return False
            time.sleep(0.10)
            continue
        
        lost_count = 0
        cx, cy, bbox = d
        err_x = cx - IMG_CX
        err_y = cy - IMG_CY
        
        kp, approach_step = get_gains(bbox)
        
        is_centered = abs(err_x) < CENTER_THRESH and abs(err_y) < CENTER_THRESH
        is_close = bbox > BBOX_READY
        
        flags = ""
        if is_centered: flags += "[C]"
        if is_close: flags += "[R]"
        
        print(f"[{i}] e=({err_x:+.0f},{err_y:+.0f}) bb={bbox:.0f} {flags}")
        
        if is_centered and is_close:
            ready_count += 1
            if ready_count >= 3:
                print("\n** READY **")
                return True
        else:
            ready_count = 0
        
        # Camera-to-EE mapping (frame="ee")
        dy = err_x * kp  # X error → lateral movement
        dz = err_y * kp  # Y error → vertical movement
        
        # Approach when reasonably centered
        dx = 0
        if abs(err_x) < 100 and abs(err_y) < 150:
            if bbox < BBOX_READY:
                dx = approach_step
            elif bbox > BBOX_READY + 40:
                dx = -0.006  # Too close, back up
        
        # Clamp movements
        max_step = 0.025 if bbox < 80 else 0.015
        dx = max(-0.012, min(max_step, dx))
        dy = max(-0.015, min(0.015, dy))
        dz = max(-0.015, min(0.015, dz))
        
        if abs(dx) > 0.001 or abs(dy) > 0.001 or abs(dz) > 0.001:
            arm.move_delta(dx=dx, dy=dy, dz=dz, frame="ee")
        
        time.sleep(0.08)
    
    print("Max iterations reached")
    return False

def grasp_at_ground():
    """Two-stage descent to ground level and grasp"""
    print("\n=== GRASP ===")
    
    ee = sensors.get_ee_position()
    print(f"Current z: {ee[2]:.3f}")
    
    # Final forward approach
    arm.move_delta(dx=0.02, frame="ee", duration=0.3)
    time.sleep(0.2)
    
    ee = sensors.get_ee_position()
    current_z = ee[2]
    print(f"After approach: z={current_z:.3f}")
    
    total_drop = current_z - GROUND_Z
    print(f"Descending {total_drop:.3f}m to z={GROUND_Z}")
    
    if total_drop > 0:
        # Fast phase - 70% of drop
        halfway_z = current_z - (total_drop * 0.7)
        print(f"  Fast descent to z={halfway_z:.3f}...")
        arm.move_to_pose(z=halfway_z, duration=1.0)
        time.sleep(0.1)
        
        # Slow phase - final 30%
        print(f"  Slow descent to z={GROUND_Z:.3f}...")
        arm.move_to_pose(z=GROUND_Z, duration=1.5)
        time.sleep(0.2)
    
    ee = sensors.get_ee_position()
    lowest_z = ee[2]
    print(f"At grasp: z={lowest_z:.3f}")
    
    # Grasp
    print("Closing gripper...")
    try:
        gripper.grasp(force=100)
    except:
        gripper.close(force=100)
    time.sleep(0.4)
    
    s = gripper.get_state()
    print(f"Gripper: pos={s['position']}")
    
    if s['position'] > 250:
        print("MISS - gripper too open")
        gripper.open()
        return False
    
    # Lift
    print("Lifting...")
    arm.move_delta(dz=0.25, frame="base", duration=0.8)
    time.sleep(0.3)
    
    s = gripper.get_state()
    success = s['object_detected'] or s['position'] < 245
    print(f"Result: {'SUCCESS!' if success else 'Dropped?'}")
    return success

def pick_up_object(target="banana"):
    """Main entry point: pick up target object from ground"""
    print(f"=== Pick Up Object: {target} ===")
    print(f"Ground level: z={GROUND_Z}\n")
    
    ee = sensors.get_ee_position()
    print(f"Start z={ee[2]:.3f}\n")
    
    if run_ibvs(target):
        success = grasp_at_ground()
    else:
        success = False
    
    print(f"\n=== {'SUCCESS 🍌' if success else 'FAIL'} ===")
    return success

# Run if executed directly
if __name__ == "__main__":
    pick_up_object("banana")
