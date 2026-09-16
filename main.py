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
        
        # 3. Workflow sequence (with clearly separated functions)
        chassis_ctrl.setup_csv_headers()            # Prepare the CSV file.
        chassis_ctrl.start_sensors()   # sensor data reception
        
        chassis_ctrl.test_movement()        # Move (or simply call .move_forward())
        
        chassis_ctrl.stop_sensors()    # Stop receiving sensor data.
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ep_robot.close()
        print("Robot connection closed successfully.")

if __name__ == '__main__':
    main()