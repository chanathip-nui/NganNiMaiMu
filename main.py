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
                    if m["color"] == d["color"] and get_distance(m["center"], d["center"]) < 120:
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

        # จัดเรียงลำดับการยิงจากซ้ายไปขวา (ตามแนวแกน X)
        memorized_targets.sort(key=lambda t: t["center"][0])

        # ==========================================================
        # PHASE 2: ยิงไล่ทีละเป้าหมายตามรายการที่จำไว้ (ไม่สแกนใหม่)
        # ==========================================================
        print("\n[PHASE 2] Starting Sequential Elimination...")
        total_targets = len(memorized_targets)

        for idx, target in enumerate(memorized_targets):
            print(f"\n>> [{idx + 1}/{total_targets}] Targeting: {target['color']} {target['shape']}")

            target_shot = False
            aim_start_time = time.time()
            current_target_pos = target["center"]

            # วนลูปเล็งและยิงเฉพาะเป้านี้ (Timeout 6 วินาทีต่อเป้า)
            while time.time() - aim_start_time < 6.0:
                frame = ep_robot.camera.read_cv2_image(strategy="newest")
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                frame, detected_targets = detector.detect(frame, draw=True)

                # หาเป้าที่มีสีตรงกับเป้าหมายปัจจุบัน
                matched_candidates = [t for t in detected_targets if t["color"] == target["color"]]

                if matched_candidates:
                    # เลือกตัวที่ใกล้ตำแหน่งเดิมที่สุด
                    best_match = min(matched_candidates, key=lambda t: get_distance(t["center"], current_target_pos))
                    current_target_pos = best_match["center"]

                    # ตีกรอบสีขาวแสดงเป้าหมายที่กำลังยิง
                    bx, by, bw, bh = best_match["bbox"]
                    cv2.rectangle(frame, (bx-2, by-2), (bx+bw+2, by+bh+2), (255, 255, 255), 2)
                    cv2.putText(frame, f"FIRING AT: {target['color']}", (bx, max(15, by - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # สั่ง PID เล็งและยิง
                    is_fired = gimbal_ctrl.aim_and_shoot(current_target_pos, w, h)
                    if is_fired:
                        print(f">> Target {target['color']} ELIMINATED!")
                        target_shot = True
                        time.sleep(0.8) # หน่วงเวลาหลังยิง
                        break
                else:
                    # หากเป้าหลุดเฟรมระหว่างกิมบอลหมุน ให้หยุดกิมบอลรอเฟรมถัดไป
                    ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

                # แสดงสถานะบนหน้าจอ
                cv2.putText(frame, f"Eliminating Target {idx + 1}/{total_targets} ({target['color']})", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Auto Target Acquisition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if not target_shot:
                print(f">> Timeout aiming at {target['color']}, moving to next target.")

        print("\nAll memorized targets have been processed successfully!")

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