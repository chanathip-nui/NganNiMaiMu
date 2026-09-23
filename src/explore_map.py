# src/explore_map.py
# -*- coding: utf-8 -*-
"""
DFS Grid Exploration for RoboMaster EP

Explore a 4x5 Grid (each cell 60cm) using:
- ToF Sensor on Gimbal: rotate gimbal to scan 3 directions (front/left/right)
- Chassis odometry: move forward 1 cell, turn 90/180 degrees
- DFS Algorithm: explore until all reachable cells are visited

No IR sensor, no goal - pure 100% coverage exploration
"""

import csv
import os
import time
import threading
from datetime import datetime


# ============================================================
# Headings: 0=North(+Y), 1=East(+X), 2=South(-Y), 3=West(-X)
# ============================================================
HEADING_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}
HEADING_DEGREES = {0: 0, 1: 90, 2: 180, 3: 270}

# Delta (dx, dy) for each heading - next cell when moving forward
HEADING_DELTA = {
    0: (0, 1),   # North -> y+1
    1: (1, 0),   # East  -> x+1
    2: (0, -1),  # South -> y-1
    3: (-1, 0),  # West  -> x-1
}


class GridExplorer:
    """Explore Grid Map using DFS + ToF sensor on Gimbal"""

    def __init__(self, ep_robot, config, gimbal_ctrl=None):
        """
        Args:
            ep_robot: RoboMaster EP robot instance (initialized)
            config: dict from settings.yaml
            gimbal_ctrl: optional GimbalController instance
        """
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_gimbal = ep_robot.gimbal
        self.ep_sensor = ep_robot.sensor
        self.config = config

        if gimbal_ctrl is not None:
            self.gimbal_ctrl = gimbal_ctrl
        else:
            from .gimbal import GimbalController
            self.gimbal_ctrl = GimbalController(ep_robot, config)

        # --- Grid Map config ---
        grid_cfg = config["grid_map"]
        self.max_x = grid_cfg["max_x"]          # 4
        self.max_y = grid_cfg["max_y"]          # 5
        self.cell_size = grid_cfg["cell_size"]  # 0.6m

        self.start_x = grid_cfg.get("start_x", 0)
        self.start_y = grid_cfg.get("start_y", 0)
        self.start_heading = grid_cfg.get("start_heading", 0)

        # --- Exploration config ---
        exp_cfg = config.get("exploration", {})
        self.wall_threshold_mm = exp_cfg.get("wall_threshold_mm", 400)
        self.scan_settle_time = exp_cfg.get("scan_settle_time", 0.5)
        self.tof_samples = exp_cfg.get("tof_samples", 3)
        self.tof_read_interval = exp_cfg.get("tof_read_interval", 0.1)
        self.move_speed = exp_cfg.get("move_speed", 0.5)
        self.turn_speed = exp_cfg.get("turn_speed", 60)

        # --- Data collection config ---
        data_cfg = config.get("data_collection", {})
        raw_data_dir = data_cfg.get("data_dir", "data/raw/run1")
        if not os.path.isabs(raw_data_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, raw_data_dir)
        else:
            self.data_dir = raw_data_dir

        # --- State ---
        self.current_tof_dist_mm = 0
        self._tof_lock = threading.Lock()

        # --- Visited grid (2D boolean array) ---
        self.visited = [[False] * self.max_y for _ in range(self.max_x)]

        # --- Wall map: store detected walls ---
        # wall_map[(x,y)] = {heading: True/False}  True = wall exists
        self.wall_map = {}

        # --- Step counter ---
        self.step_count = 0

    # ================================================================
    # ToF Sensor Handling
    # ================================================================

    def _tof_callback(self, sub_info):
        """Callback for ToF distance subscription

        sub_info format: [tof1, tof2, tof3, tof4] in mm
        """
        with self._tof_lock:
            if isinstance(sub_info, (list, tuple)):
                # RoboMaster EP sends [tof1, tof2, tof3, tof4]
                # Find the first positive distance reading (usually sensor 1)
                valid = [x for x in sub_info if isinstance(x, (int, float)) and x > 0]
                if valid:
                    self.current_tof_dist_mm = float(valid[0])
                elif len(sub_info) > 0 and isinstance(sub_info[0], (int, float)):
                    self.current_tof_dist_mm = float(sub_info[0])
            elif isinstance(sub_info, (int, float)):
                self.current_tof_dist_mm = float(sub_info)

    def _read_tof_averaged(self):
        """Read ToF multiple times and average to reduce noise

        Returns:
            float: averaged ToF distance (mm)
        """
        readings = []
        for _ in range(self.tof_samples):
            time.sleep(self.tof_read_interval)
            with self._tof_lock:
                if self.current_tof_dist_mm > 0:
                    readings.append(self.current_tof_dist_mm)

        if readings:
            return sum(readings) / len(readings)
        # Fallback to current value or 9999 if no positive readings
        with self._tof_lock:
            return self.current_tof_dist_mm if self.current_tof_dist_mm > 0 else 9999

    def is_wall(self, distance_mm):
        """Check if ToF distance < threshold = wall

        Args:
            distance_mm: ToF distance (mm)

        Returns:
            bool: True = wall detected
        """
        # Distances <= 0 indicate uninitialized or out-of-range/error -> not a wall
        if distance_mm <= 0 or distance_mm >= 9000:
            return False
        return distance_mm < self.wall_threshold_mm

    # ================================================================
    # Gimbal Scanning - rotate gimbal to scan 3 directions
    # ================================================================

    def scan_with_gimbal(self):
        """Rotate gimbal to scan ToF in 3 directions (front/left/right)

        Steps:
        0. Reset gimbal to center (0, 0) and reset PID
        1. Gimbal yaw = 0    -> read ToF -> front distance
        2. Gimbal yaw = -90  -> read ToF -> left distance
        3. Gimbal yaw = +90  -> read ToF -> right distance
        4. Reset gimbal to center (0, 0)

        Returns:
            dict: {"front": mm, "left": mm, "right": mm}
        """
        # --- 0. Reset gimbal ก่อน scan ทุกครั้ง ---
        self.gimbal_ctrl.reset_gimbal()
        time.sleep(self.scan_settle_time)

        result = {}

        # --- 1. Scan front (yaw = 0) ---
        self.ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
        time.sleep(self.scan_settle_time)
        result["front"] = self._read_tof_averaged()

        # --- 2. Scan left (yaw = -90) ---
        self.ep_gimbal.moveto(pitch=0, yaw=-90).wait_for_completed()
        time.sleep(self.scan_settle_time)
        result["left"] = self._read_tof_averaged()

        # --- 3. Scan right (yaw = +90) ---
        self.ep_gimbal.moveto(pitch=0, yaw=90).wait_for_completed()
        time.sleep(self.scan_settle_time)
        result["right"] = self._read_tof_averaged()

        # --- 4. Recenter & reset gimbal ---
        self.gimbal_ctrl.reset_gimbal()

        return result

    # ================================================================
    # Movement - move forward / turn
    # ================================================================

    def move_forward_one_cell(self):
        """Move forward 1 cell (cell_size = 0.6m)"""
        self.ep_chassis.move(
            x=self.cell_size, y=0, z=0,
            xy_speed=self.move_speed
        ).wait_for_completed()

    def turn_right_90(self):
        """Turn right 90 degrees"""
        self.ep_chassis.move(
            x=0, y=0, z=-90,
            z_speed=self.turn_speed
        ).wait_for_completed()

    def turn_left_90(self):
        """Turn left 90 degrees"""
        self.ep_chassis.move(
            x=0, y=0, z=90,
            z_speed=self.turn_speed
        ).wait_for_completed()

    def turn_180(self):
        """Turn 180 degrees"""
        self.ep_chassis.move(
            x=0, y=0, z=180,
            z_speed=self.turn_speed
        ).wait_for_completed()

    def _turn_to_heading(self, current_heading, target_heading):
        """Turn from current heading to target heading

        Args:
            current_heading: current direction (0-3)
            target_heading: target direction (0-3)

        Returns:
            int: new heading after turning
        """
        diff = (target_heading - current_heading) % 4
        if diff == 0:
            pass  # no turn needed
        elif diff == 1:
            self.turn_right_90()
        elif diff == 2:
            self.turn_180()
        elif diff == 3:
            self.turn_left_90()
        return target_heading

    # ================================================================
    # Grid Helpers
    # ================================================================

    def _in_bounds(self, x, y):
        """Check if (x, y) is within grid bounds"""
        return 0 <= x < self.max_x and 0 <= y < self.max_y

    def _get_neighbor(self, x, y, heading):
        """Calculate next cell position in given heading

        Args:
            x, y: current position
            heading: direction to move (0-3)

        Returns:
            tuple: (nx, ny) or None if out of bounds
        """
        dx, dy = HEADING_DELTA[heading]
        nx, ny = x + dx, y + dy
        if self._in_bounds(nx, ny):
            return (nx, ny)
        return None

    def _relative_to_absolute_heading(self, robot_heading, relative_dir):
        """Convert relative direction (front/left/right) to absolute heading (0-3)

        Args:
            robot_heading: robot facing direction (0-3)
            relative_dir: "front", "left", or "right"

        Returns:
            int: absolute heading (0-3)
        """
        if relative_dir == "front":
            return robot_heading
        elif relative_dir == "right":
            return (robot_heading + 1) % 4
        elif relative_dir == "left":
            return (robot_heading + 3) % 4  # +3 = -1 mod 4
        return robot_heading

    # ================================================================
    # Main DFS Exploration
    # ================================================================

    def explore_and_map_all(self):
        """DFS algorithm to map 4x5 Grid - explore 100%

        Uses ToF sensor on Gimbal to scan 3 directions instead of IR
        Logs all steps and sensor readings to CSV file
        """
        print("=" * 60)
        print("  Start DFS Mapping")
        print("  Grid: {}x{} | Cell: {}m".format(
            self.max_x, self.max_y, self.cell_size))
        print("  Wall Threshold: {}mm".format(self.wall_threshold_mm))
        print("=" * 60)

        # =====================================================
        # 1. Prepare CSV file
        # =====================================================
        os.makedirs(self.data_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")

        files_cfg = self.config.get("data_collection", {}).get("files", {})
        filename_key = files_cfg.get("exploration", "exploration_grid_data")
        csv_path = os.path.join(
            self.data_dir, "log_{}_{}.csv".format(date_str, filename_key)
        )

        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow([
            "unix_timestamp",
            "step",
            "grid_x", "grid_y",
            "real_x_m", "real_y_m",
            "heading", "heading_deg",
            "tof_front_mm", "tof_left_mm", "tof_right_mm",
            "wall_front", "wall_left", "wall_right",
            "action",
        ])

        # =====================================================
        # 2. Subscribe ToF sensor
        # =====================================================
        freq_dist = self.config["data_collection"]["frequencies"]["distance"]
        self.ep_sensor.sub_distance(
            freq=freq_dist, callback=self._tof_callback
        )
        print("Connecting ToF sensor stream...")
        start_wait = time.time()
        while time.time() - start_wait < 3.0:
            with self._tof_lock:
                if self.current_tof_dist_mm > 0:
                    break
            time.sleep(0.1)

        with self._tof_lock:
            init_dist = self.current_tof_dist_mm

        if init_dist > 0:
            print("  ✓ ToF sensor active (reading: {:.0f} mm)".format(init_dist))
        else:
            print("  ⚠ WARNING: ToF sensor reading is 0 mm! Check sensor connection/cable.")

        # =====================================================
        # 3. Helper: log 1 step to CSV
        # =====================================================
        def log_step(gx, gy, hdg, scan_result, action):
            """Log 1 step data to CSV"""
            self.step_count += 1
            t = time.time()

            front_mm = scan_result.get("front", 0)
            left_mm = scan_result.get("left", 0)
            right_mm = scan_result.get("right", 0)

            writer.writerow([
                "{:.3f}".format(t),
                self.step_count,
                gx, gy,
                "{:.2f}".format(gx * self.cell_size),
                "{:.2f}".format(gy * self.cell_size),
                HEADING_NAMES.get(hdg, "?"),
                HEADING_DEGREES.get(hdg, 0),
                "{:.0f}".format(front_mm),
                "{:.0f}".format(left_mm),
                "{:.0f}".format(right_mm),
                1 if self.is_wall(front_mm) else 0,
                1 if self.is_wall(left_mm) else 0,
                1 if self.is_wall(right_mm) else 0,
                action,
            ])
            csv_file.flush()

            print(
                "  Step {:3d} | ({},{}) {} | "
                "F:{:.0f} L:{:.0f} R:{:.0f} | {}".format(
                    self.step_count,
                    gx, gy, HEADING_NAMES[hdg],
                    front_mm, left_mm, right_mm,
                    action
                )
            )

        # =====================================================
        # 4. DFS Exploration (Iterative with explicit stack)
        # =====================================================
        x = self.start_x
        y = self.start_y
        heading = self.start_heading

        self.visited[x][y] = True
        total_cells = self.max_x * self.max_y
        visited_count = 1

        # Stack for backtracking: [(x, y, heading)]
        path_stack = [(x, y, heading)]

        print("\n  Start: ({},{}) heading={}".format(
            x, y, HEADING_NAMES[heading]))
        print("  Total cells: {}".format(total_cells))
        print("-" * 60)

        try:
            while path_stack:
                if visited_count >= total_cells:
                    break

                # --- Scan 3 directions with gimbal ---
                scan = self.scan_with_gimbal()

                # --- Record wall map ---
                for rel_dir in ["front", "left", "right"]:
                    abs_h = self._relative_to_absolute_heading(
                        heading, rel_dir)
                    key = (x, y)
                    if key not in self.wall_map:
                        self.wall_map[key] = {}
                    self.wall_map[key][abs_h] = self.is_wall(scan[rel_dir])

                log_step(x, y, heading, scan, "SCAN")

                # --- Find next unvisited, wall-free cell ---
                # Priority: front -> right -> left (right-hand rule variant)
                moved = False

                for rel_dir in ["front", "right", "left"]:
                    abs_h = self._relative_to_absolute_heading(
                        heading, rel_dir)
                    neighbor = self._get_neighbor(x, y, abs_h)

                    if (
                        neighbor is not None
                        and not self.visited[neighbor[0]][neighbor[1]]
                        and not self.is_wall(scan[rel_dir])
                    ):
                        # Turn to face that direction
                        heading = self._turn_to_heading(heading, abs_h)

                        # Move forward 1 cell
                        self.move_forward_one_cell()

                        # Update position
                        x, y = neighbor
                        self.visited[x][y] = True
                        visited_count += 1

                        path_stack.append((x, y, heading))

                        log_step(x, y, heading, scan, "MOVE_FORWARD")

                        progress = (visited_count / total_cells) * 100
                        print(
                            "  >>> Visited: {}/{} ({:.1f}%)".format(
                                visited_count, total_cells, progress
                            )
                        )

                        # ถ้า visit ครบทุกช่องแล้ว ให้หยุดทันที ไม่ต้องเดิน backtrack
                        if visited_count >= total_cells:
                            print(
                                "\n  >>> Visited all {}/{} cells (100.0%)! Stopping immediately!".format(
                                    visited_count, total_cells
                                )
                            )
                            log_step(x, y, heading, scan, "COMPLETE_ALL_VISITED")
                            break

                        moved = True
                        break

                if visited_count >= total_cells:
                    break

                if not moved:
                    # --- No way forward -> Backtrack ---
                    path_stack.pop()  # remove current cell from stack

                    if path_stack:
                        prev_x, prev_y, prev_heading = path_stack[-1]

                        # Calculate direction to go back
                        dx = prev_x - x
                        dy = prev_y - y

                        # Find heading back
                        back_heading = None
                        for h, (ddx, ddy) in HEADING_DELTA.items():
                            if ddx == dx and ddy == dy:
                                back_heading = h
                                break

                        if back_heading is not None:
                            heading = self._turn_to_heading(
                                heading, back_heading)
                            self.move_forward_one_cell()
                            x, y = prev_x, prev_y
                            # Restore previous heading physically
                            heading = self._turn_to_heading(
                                heading, prev_heading)

                            log_step(x, y, heading, scan, "BACKTRACK")
                        else:
                            print("  !!! ERROR: Cannot calculate "
                                  "back heading")
                            break
                    else:
                        # Stack empty = exploration complete
                        log_step(x, y, heading, scan, "COMPLETE")

            # =====================================================
            # 5. Summary
            # =====================================================
            coverage = (visited_count / total_cells) * 100
            print("\n" + "=" * 60)
            print("  Exploration Complete!")
            print("  Visited: {}/{} ({:.1f}%)".format(
                visited_count, total_cells, coverage))
            print("  Total Steps: {}".format(self.step_count))
            print("  CSV saved: {}".format(csv_path))
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n  --> Exploration cancelled by user")
            self.ep_chassis.drive_speed(x=0, y=0, z=0)

        finally:
            # Cleanup
            csv_file.close()
            try:
                self.ep_sensor.unsub_distance()
            except Exception:
                pass
            try:
                self.gimbal_ctrl.reset_gimbal()
            except Exception:
                pass

        return {
            "visited_count": visited_count,
            "total_cells": total_cells,
            "coverage_pct": (visited_count / total_cells) * 100,
            "steps": self.step_count,
            "csv_path": csv_path,
            "wall_map": self.wall_map,
            "visited_grid": self.visited,
        }

    # ================================================================
    # Utility: Print grid map to console
    # ================================================================

    def print_grid(self):
        """Print visited grid to console"""
        print("\n  Grid Map ({}x{}):".format(self.max_x, self.max_y))
        border = "  " + "-" * (self.max_x * 4 + 1)
        print(border)
        # Display from highest Y down
        for y in range(self.max_y - 1, -1, -1):
            row = "  |"
            for x in range(self.max_x):
                if self.visited[x][y]:
                    row += " V |"
                else:
                    row += "   |"
            print(row)
            print(border)
        # X axis labels
        labels = "   "
        for x in range(self.max_x):
            labels += " {}  ".format(x)
        print(labels)