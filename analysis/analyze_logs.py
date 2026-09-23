# analysis/analyze_logs.py
"""
Post-mission analysis script for SLAM data.

Loads CSV logs from a completed SLAM run and generates:
1. Trajectory plot (robot path on the map)
2. Map visualization
3. Coverage & accuracy statistics
4. Scan density heatmap

Usage:
    python analysis/analyze_logs.py --data-dir data/raw/run1
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
import json


def load_csv(filepath):
    """Load a CSV file into a list of dicts."""
    if not os.path.exists(filepath):
        print(f"  [SKIP] File not found: {filepath}")
        return []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  [OK] Loaded {len(rows)} rows from {os.path.basename(filepath)}")
    return rows


def find_latest_file(data_dir, prefix):
    """Find the latest file matching a prefix in the data dir."""
    matches = [f for f in os.listdir(data_dir) if f.startswith(prefix)]
    if not matches:
        return None
    matches.sort(reverse=True)  # latest first (timestamp in name)
    return os.path.join(data_dir, matches[0])


def plot_trajectory(trajectory_data, map_img_path, output_path, config=None):
    """
    Draw robot trajectory on top of the saved map image.

    Args:
        trajectory_data: list of dicts with robot_x, robot_y, robot_yaw
        map_img_path: path to the saved map image
        output_path: where to save the trajectory plot
    """
    if not trajectory_data:
        print("  No trajectory data to plot")
        return

    # Load map image if available, else create blank
    if map_img_path and os.path.exists(map_img_path):
        canvas = cv2.imread(map_img_path)
    else:
        # Create a blank canvas
        canvas = np.full((400, 320, 3), 200, dtype=np.uint8)

    h, w = canvas.shape[:2]

    # Extract trajectory points
    points = []
    for row in trajectory_data:
        x = float(row["robot_x"])
        y = float(row["robot_y"])
        points.append((x, y))

    if not points:
        return

    # Determine bounds for scaling
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = 0.5  # meters

    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin

    def to_pixel(x, y):
        px = int((x - x_min) / (x_max - x_min) * (w - 20) + 10)
        py = int((y - y_min) / (y_max - y_min) * (h - 20) + 10)
        return px, py

    # Draw trajectory
    pixel_points = [to_pixel(x, y) for x, y in points]
    for i in range(1, len(pixel_points)):
        cv2.line(canvas, pixel_points[i-1], pixel_points[i],
                 (255, 120, 0), 2, cv2.LINE_AA)

    # Mark start (green) and end (red)
    cv2.drawMarker(canvas, pixel_points[0], (0, 200, 0),
                   cv2.MARKER_DIAMOND, 15, 2)
    cv2.drawMarker(canvas, pixel_points[-1], (0, 0, 255),
                   cv2.MARKER_STAR, 15, 2)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    sx, sy = points[0]
    ex, ey = points[-1]
    cv2.putText(canvas, f"Start ({sx:.2f}, {sy:.2f})",
                (pixel_points[0][0] + 10, pixel_points[0][1] - 10),
                font, 0.35, (0, 150, 0), 1)
    cv2.putText(canvas, f"End ({ex:.2f}, {ey:.2f})",
                (pixel_points[-1][0] + 10, pixel_points[-1][1] - 10),
                font, 0.35, (0, 0, 200), 1)

    cv2.imwrite(output_path, canvas)
    print(f"  Trajectory plot saved: {output_path}")


def print_summary(data_dir):
    """Print summary from the summary text file."""
    summaries = [f for f in os.listdir(data_dir) if f.endswith("_summary.txt")]
    if summaries:
        summaries.sort(reverse=True)
        path = os.path.join(data_dir, summaries[0])
        with open(path, "r", encoding="utf-8") as f:
            print(f.read())

    # Also print JSON evaluation if available
    evals = [f for f in os.listdir(data_dir) if f.endswith("_evaluation.json")]
    if evals:
        evals.sort(reverse=True)
        path = os.path.join(data_dir, evals[0])
        with open(path, "r") as f:
            results = json.load(f)
        print("Evaluation Results (JSON):")
        for k, v in results.items():
            print(f"  {k}: {v}")


def analyze(data_dir):
    """Run full post-mission analysis."""
    print(f"\n{'='*50}")
    print(f"  SLAM Post-Mission Analysis")
    print(f"  Data directory: {data_dir}")
    print(f"{'='*50}\n")

    if not os.path.exists(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        return

    # 1. Load trajectory data
    print("Loading data files...")
    traj_file = find_latest_file(data_dir, "slam_") 
    traj_files = [f for f in os.listdir(data_dir) if "trajectory" in f]
    traj_data = []
    if traj_files:
        traj_files.sort(reverse=True)
        traj_data = load_csv(os.path.join(data_dir, traj_files[0]))

    # 2. Load scan data
    scan_files = [f for f in os.listdir(data_dir) if "scan_log" in f]
    scan_data = []
    if scan_files:
        scan_files.sort(reverse=True)
        scan_data = load_csv(os.path.join(data_dir, scan_files[0]))

    # 3. Load decision data
    decision_files = [f for f in os.listdir(data_dir) if "decisions" in f]
    decision_data = []
    if decision_files:
        decision_files.sort(reverse=True)
        decision_data = load_csv(os.path.join(data_dir, decision_files[0]))

    # 4. Find map image
    map_files = [f for f in os.listdir(data_dir) if "map_final" in f]
    map_img_path = None
    if map_files:
        map_files.sort(reverse=True)
        map_img_path = os.path.join(data_dir, map_files[0])
        print(f"  [OK] Map image: {map_files[0]}")

    # 5. Generate trajectory plot
    print("\nGenerating trajectory plot...")
    traj_output = os.path.join(data_dir, "analysis_trajectory.png")
    plot_trajectory(traj_data, map_img_path, traj_output)

    # 6. Print statistics
    print(f"\n{'='*50}")
    print("  Statistics")
    print(f"{'='*50}")
    print(f"  Total scan readings : {len(scan_data)}")
    print(f"  Trajectory points   : {len(traj_data)}")
    print(f"  Decisions made      : {len(decision_data)}")

    if decision_data:
        actions = {}
        for row in decision_data:
            a = row.get("action", "UNKNOWN")
            actions[a] = actions.get(a, 0) + 1
        print(f"\n  Decision breakdown:")
        for action, count in sorted(actions.items()):
            print(f"    {action}: {count}")

    # 7. Print mission summary
    print(f"\n{'='*50}")
    print_summary(data_dir)

    print(f"\n✅ Analysis complete. Output saved to: {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze SLAM mission logs")
    parser.add_argument(
        "--data-dir", type=str,
        default="data/raw/run1",
        help="Path to the data directory (default: data/raw/run1)")
    args = parser.parse_args()

    # Handle relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(base_dir, data_dir)

    analyze(data_dir)
