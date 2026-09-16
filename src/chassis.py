import csv
import time
import os
import threading
from datetime import datetime


class ChassisController:
    def __init__(self, ep_robot, config):
        # 1. Connect to the robot unit, chassis, and sensors.
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_sensor = ep_robot.sensor  # <-- NEW: Initialize the sensor module

        # 2. Retrieve values from the configuration.
        self.data_dir = config["data_collection"]["data_dir"]
        self.buffer_time = config["data_collection"]["buffer_time"]

        # Extract the frequency of each sensor.
        self.freq_pos = config["data_collection"]["frequencies"]["position"]
        self.freq_att = config["data_collection"]["frequencies"]["attitude"]
        self.freq_imu = config["data_collection"]["frequencies"]["imu"]
        self.freq_esc = config["data_collection"]["frequencies"]["esc"]
        self.freq_dist = config["data_collection"]["frequencies"][
            "distance"
        ]  # <-- NEW: Distance frequency

        # Retrieve motion-related values.
        self.default_speed = config["movement"]["xy_speed"]
        self.default_distance = config["movement"]["distance"]
        self.default_z_speed = config["movement"]["z_speed"]
        self.default_angle = config["movement"]["angle"]

        # 3. Create a folder to store the files.
        os.makedirs(self.data_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")

        # Combine the date to form filenames.
        pos_name = (
            f"log_{date_str}_{config['data_collection']['files']['position']}.csv"
        )
        att_name = (
            f"log_{date_str}_{config['data_collection']['files']['attitude']}.csv"
        )
        imu_name = f"log_{date_str}_{config['data_collection']['files']['imu']}.csv"
        esc_name = f"log_{date_str}_{config['data_collection']['files']['esc']}.csv"
        dist_name = f"log_{date_str}_{config['data_collection']['files']['distance']}.csv"  # <-- NEW: Distance filename

        # 4. Name the file paths for the CSV files.
        self.pos_file = os.path.join(self.data_dir, pos_name)
        self.att_file = os.path.join(self.data_dir, att_name)
        self.imu_file = os.path.join(self.data_dir, imu_name)
        self.esc_file = os.path.join(self.data_dir, esc_name)
        self.dist_file = os.path.join(
            self.data_dir, dist_name
        )  # <-- NEW: Distance file path


    # =========================================================
    # Part 1: Function for writing a CSV file and receiving a return value
    # =========================================================
    def save_to_csv(self, filename, data):
        """A central function for appending data to a CSV file."""
        current_time = time.time()
        with open(filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = [current_time]
            for item in data:
                if isinstance(item, (list, tuple)):
                    row.extend(item)
                else:
                    row.append(item)
            writer.writerow(row)

    def handle_position(self, data):
        self.save_to_csv(self.pos_file, data)

    def handle_attitude(self, data):
        self.save_to_csv(self.att_file, data)

    def handle_imu(self, data):
        self.save_to_csv(self.imu_file, data)

    def handle_esc(self, data):
        self.save_to_csv(self.esc_file, data)

    def handle_distance(self, data):
        # <-- NEW: Callback for distance data
        self.save_to_csv(self.dist_file, data)

    # =========================================================
    # Part 2: Sensor Data Collection Management Function
    # =========================================================
    def setup_csv_headers(self):
        """Create a new CSV file and write the column headers."""
        with open(self.pos_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "x", "y", "z"])

        with open(self.att_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "yaw", "pitch", "roll"])

        with open(self.imu_file, mode="w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "unix_timestamp",
                    "acc_x",
                    "acc_y",
                    "acc_z",
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                ]
            )

        with open(self.esc_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["unix_timestamp", "esc_data"])

        with open(self.dist_file, mode="w", newline="") as f:
            # <-- NEW: Write headers for the 4 ToF sensors
            csv.writer(f).writerow(["unix_timestamp", "tof1"])


        print("All CSV files have been prepared.")


    def start_sensors(self):
        """Activate all sensors according to the frequency set in the configuration."""
        print("Starting to collect sensor data...")
        self.ep_chassis.sub_position(freq=self.freq_pos, callback=self.handle_position)
        self.ep_chassis.sub_attitude(freq=self.freq_att, callback=self.handle_attitude)
        self.ep_chassis.sub_imu(freq=self.freq_imu, callback=self.handle_imu)
        self.ep_chassis.sub_esc(freq=self.freq_esc, callback=self.handle_esc)
        self.ep_sensor.sub_distance(
            freq=self.freq_dist, callback=self.handle_distance
        )  # <-- NEW: Start distance sensor


    def stop_sensors(self):
        """Disable all sensors."""
        time.sleep(self.buffer_time)
        self.ep_chassis.unsub_position()
        self.ep_chassis.unsub_attitude()
        self.ep_chassis.unsub_imu()
        self.ep_chassis.unsub_esc()
        self.ep_sensor.unsub_distance()  # <-- NEW: Stop distance sensor

        print("Data collection and saving to the file have been fully completed.")

