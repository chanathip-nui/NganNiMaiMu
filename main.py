# main.py
"""
RoboMaster EP — DFS Grid Exploration & Mapping

Main entry point:
1. Connect to robot
2. Run DFS exploration on 4x5 grid
3. Use ToF sensor on gimbal to scan walls
4. Log all data to CSV
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robomaster import robot
from src.config_loader import load_config
from src.gimbal import GimbalController
from src.explore_map import GridExplorer


def main():
    config = load_config("config/settings.yaml")

    ep_robot = robot.Robot()

    explorer = None

    try:
        print("=" * 50)
        print("  RoboMaster EP — DFS Grid Exploration")
        print("=" * 50)
        print("Connecting robot ....")
        ep_robot.initialize()
        print("Connected!\n")

        # Reset gimbal
        gimbal_ctrl = GimbalController(ep_robot, config)
        gimbal_ctrl.reset_gimbal()

        # =====================================================
        # GridExplorer manages its own ToF subscription
        # Do NOT call chassis_ctrl.start_sensors() here
        # to avoid double-subscribing to ToF
        # =====================================================
        explorer = GridExplorer(ep_robot, config, gimbal_ctrl=gimbal_ctrl)

        print("\nStart Grid Map Exploration...")
        print("Grid: {}x{}".format(explorer.max_x, explorer.max_y))
        print("Cell size: {}m".format(explorer.cell_size))
        print("Start: ({},{})".format(explorer.start_x, explorer.start_y))
        print("Heading: {}\n".format(explorer.start_heading))

        # Run DFS exploration
        result = explorer.explore_and_map_all()

        # Print grid map
        explorer.print_grid()

        # Summary
        print("\n--- Summary ---")
        print("Coverage: {:.1f}%".format(result['coverage_pct']))
        print("Steps: {}".format(result['steps']))
        print("CSV: {}".format(result['csv_path']))

    except KeyboardInterrupt:
        print("\n[Ctrl+C detected] Halting robot movement...")

    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()

    finally:
        # Stop robot movement
        try:
            ep_robot.chassis.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass
        try:
            ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception:
            pass
        try:
            ep_robot.close()
        except Exception:
            pass
        print("\nRobot connection closed successfully.")


if __name__ == '__main__':
    main()
