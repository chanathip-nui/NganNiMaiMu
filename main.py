from robomaster import robot

# main.py
import sys
import os
from robomaster import robot
import time

# Add path for import module in src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config_loader import load_config
from src.chassis import ChassisController
from src.camera import CameraController
from src.gimbal import GimbalController

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

        gimbal_ctrl.gimbal_shoot()    
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
        # 1. Zero wheel velocity immediately to prevent runaway motion
        try:
            ep_robot.chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(0.1)
        except Exception:
            pass

        # 2. Safely unregister sensor listeners and flush CSV buffers
        if chassis_ctrl:
            try:
                chassis_ctrl.stop_sensors()
            except Exception as err:
                print(f"Error while stopping sensors: {err}")

        # 3. Release hardware connection
        ep_robot.close()
        print("Robot connection closed successfully.")

if __name__ == '__main__':
    main()