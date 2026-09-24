from robomaster import robot

# main.py
import sys
import os
import cv2
import numpy as np
from robomaster import robot
import time
import math

# Add path for import module in src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config_loader import load_config
from src.chassis import ChassisController
from src.camera import CameraController
from src.gimbal import GimbalController
from src.detector import TargetDetector

def get_target_id(target, current_yaw=None):
    """
    สร้าง Unique ID สำหรับเป้าหมาย:
    - หากสี/รูปทรงไม่ซ้ำ: ใช้ f"{target['color']}_{target['shape']}"
    - หากซ้ำ: รวมองศา Gimbal Yaw เข้าไปด้วย เช่น f"{target['color']}_{target['shape']}_{round(current_yaw/10)*10}"
    """
    return f"{target['color']}_{target['shape']}"

def get_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def bounding_box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = (aw * ah) + (bw * bh) - intersection
    return intersection / union if union else 0.0


def estimate_spacing_m(targets, config):
    """Estimate physical spacing from adjacent target centers in the image."""
    if len(targets) < 2:
        targeting = config["targeting"]
        return (targeting["spacing_min_m"] + targeting["spacing_max_m"]) / 2

    ordered = sorted(targets, key=lambda target: target["center"][0])
    pixel_gaps = [
        ordered[index + 1]["center"][0] - ordered[index]["center"][0]
        for index in range(len(ordered) - 1)
    ]
    pixel_gap = float(np.median(pixel_gaps))
    targeting = config["targeting"]
    spacing_m = pixel_gap / targeting["pixels_per_meter"]
    return max(targeting["spacing_min_m"], min(targeting["spacing_max_m"], spacing_m))


def is_required_target(target, targeting):
    return (
        target["color"] == targeting["required_color"]
        and target["shape"] == targeting["required_shape"]
    )

def main():
    # Download file setting form settings.yaml
    config = load_config("config/settings.yaml")
    
    ep_robot = robot.Robot()
    
    try:
        print("Connecting robot ....")
        # Use connection_type from yaml
        ep_robot.initialize()
        
        # 2. Initialize the chassis control class by passing in the configuration.
        chassis_ctrl = ChassisController(ep_robot, config)
        camera_ctrl = CameraController(ep_robot, config)
        gimbal_ctrl = GimbalController(ep_robot, config)
        detector = TargetDetector()

        gimbal_ctrl.reset_gimbal()

        camera_ctrl.start_camera()
        time.sleep(1) # รอเฟรมแรกของกล้อง

        chassis_ctrl.setup_csv_headers()
        chassis_ctrl.start_sensors()
        
        memorized_targets = []
        scan_duration = 2.0  # สแกนต่อเนื่อง 2 วินาที เพื่อเก็บภาพให้ครบทุกเป้า
        scan_start = time.time()

        print("System Ready. Scanning for targets...")

        while time.time() - scan_start < scan_duration:
            frame = ep_robot.camera.read_cv2_image(strategy="newest")
            if frame is None:
                continue

            frame, detected = detector.detect(frame, draw=True)

            for d in detected:
                # ตรวจสอบว่าเป้านี้ถูกบันทึกไปแล้วหรือยัง (เช็คจากสีและตำแหน่งใกล้เคียง)
                is_existing = False
                for m in memorized_targets:
                    if (
                        m["color"] == d["color"]
                        and m["shape"] == d["shape"]
                        and bounding_box_iou(m["bbox"], d["bbox"]) > 0.35
                    ):
                        # อัปเดตพิกัดให้แม่นยำขึ้น
                        m["center"] = d["center"]
                        m["bbox"] = d["bbox"]
                        is_existing = True
                        break

                if not is_existing:
                    memorized_targets.append({
                        "id": f"{d['color']}_{d['shape']}",
                        "color": d["color"],
                        "shape": d["shape"],
                        "center": d["center"],
                        "bbox": d["bbox"]
                    })

            # แสดง UI ระหว่างสแกน
            cv2.putText(frame, f"SCANNING... Targets Found: {len(memorized_targets)}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Auto Target Acquisition", frame)
            cv2.waitKey(1)

        print(f"\n>> Scan Completed! Found {len(memorized_targets)} targets:")
        for idx, t in enumerate(memorized_targets):
            print(f"   [{idx + 1}] Color: {t['color']}, Shape: {t['shape']}, Pos: {t['center']}")

        if not memorized_targets:
            print("No targets detected. Exiting...")
            return

        targeting = config["targeting"]
        spacing_m = estimate_spacing_m(memorized_targets, config)
        print(f">> Estimated target spacing: {spacing_m:.3f} m")

        if not targeting["min_targets"] <= len(memorized_targets) <= targeting["max_targets"]:
            print(
                f">> Warning: expected {targeting['min_targets']}-{targeting['max_targets']} "
                f"targets, found {len(memorized_targets)}."
            )

        # The image X axis represents the lateral chassis Y travel in this setup.
        memorized_targets.sort(key=lambda t: t["center"][0])

        # ==========================================================
        # PHASE 2: ตรวจเป้าที่อยู่หน้าหุ่น ยิงเฉพาะเป้าที่กำหนด แล้วเลื่อนไปช่องถัดไป
        # ==========================================================
        print(
            f"\n[PHASE 2] Target filter: {targeting['required_color']} "
            f"{targeting['required_shape']}"
        )
        # The initial scan sees only the first part of the row. Keep walking for
        # the configured row length and detect the target in front at every slot.
        total_targets = targeting["max_targets"]

        for idx in range(total_targets):
            print(f"\n>> [{idx + 1}/{total_targets}] Scanning target in front of robot")
            target_shot = False
            matching_frames = 0
            aim_start_time = time.time()

            # Re-detect on every slot so a neighboring target cannot be selected by memory.
            while time.time() - aim_start_time < 6.0:
                frame = ep_robot.camera.read_cv2_image(strategy="newest")
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                frame, detected_targets = detector.detect(frame, draw=True)

                # Only the target closest to the image center is considered "in front".
                matched_candidates = [
                    target for target in detected_targets
                    if abs(target["center"][0] - (w / 2)) < w * 0.20
                ]
                key = -1

                if matched_candidates:
                    best_match = min(
                        matched_candidates,
                        key=lambda target: abs(target["center"][0] - (w / 2)),
                    )
                    bx, by, bw, bh = best_match["bbox"]
                    box_color = (0, 255, 0) if is_required_target(best_match, targeting) else (0, 165, 255)
                    cv2.rectangle(frame, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), box_color, 2)
                    cv2.putText(frame, f"FRONT: {best_match['color']} {best_match['shape']}", (bx, max(15, by - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    cv2.putText(frame, f"Slot {idx + 1}/{total_targets}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Auto Target Acquisition", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if is_required_target(best_match, targeting):
                        matching_frames += 1
                        is_fired = gimbal_ctrl.aim_and_shoot(
                            best_match["center"],
                            w,
                            h,
                            fire_enabled=matching_frames >= 3,
                        )
                        if is_fired:
                            print(">> Required target eliminated!")
                            target_shot = True
                            time.sleep(0.8)
                            break
                    else:
                        matching_frames = 0
                        ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                        print(">> Front target does not match; skip shooting.")
                        break
                else:
                    matching_frames = 0
                    ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                    cv2.putText(frame, f"Slot {idx + 1}/{total_targets}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Auto Target Acquisition", frame)
                    key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            if not target_shot:
                print(">> No required target in front; moving to next slot.")

            if idx < total_targets - 1:
                chassis_ctrl.move_y(spacing_m * targeting["y_direction"])
                time.sleep(0.3)

        print(f"\nCompleted all {total_targets} target slots.")

        #camera
        # camera_ctrl.start_camera()
        # camera_ctrl.capture()
        # camera_ctrl.stop_camera()
        
        #sensor
        # chassis_ctrl.setup_csv_headers()            # Prepare the CSV file.
        # chassis_ctrl.start_sensors()   # sensor data reception        
        # chassis_ctrl.stop_sensors()    # Stop receiving sensor data.
        
    except KeyboardInterrupt:
        print("\n[Ctrl+C detected] Halting robot movement...")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        chassis_ctrl.stop_sensors()
        ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        camera_ctrl.stop_camera()
        ep_robot.close()
        print("Robot connection closed successfully.")

if __name__ == '__main__':
    main()