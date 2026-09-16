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
        self.ep_sensor_adaptor = ep_robot.sensor_adaptor

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
        ir_name = f"log_{date_str}_{config['data_collection']['files']['infrared']}.csv"

        # 4. Name the file paths for the CSV files.
        self.pos_file = os.path.join(self.data_dir, pos_name)
        self.att_file = os.path.join(self.data_dir, att_name)
        self.imu_file = os.path.join(self.data_dir, imu_name)
        self.esc_file = os.path.join(self.data_dir, esc_name)
        self.dist_file = os.path.join(
            self.data_dir, dist_name
        )  # <-- NEW: Distance file path
        self.ir_file = os.path.join(self.data_dir, ir_name)

        # Variables for managing the IR sensor background thread
        self.is_logging_ir = False
        self.ir_thread = None

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

        with open(self.ir_file, mode="w", newline="") as f:
            csv.writer(f).writerow(
                ["unix_timestamp", "ir1_raw", "ir1_cm", "ir2_raw", "ir2_cm"]
            )

        print("All CSV files have been prepared.")

    def read_sharp_ir_sensor_1(self):
        """Polls the ADC port and logs the raw voltage data."""
        # Read the ADC value from the sensor adapter (id=1, port=1)
        adc_value = self.ep_sensor_adaptor.get_adc(id=1, port=1)
        # print(f"Sensor adapter id1-port1 ADC is {adc_value}")

        return adc_value

    def read_sharp_ir_sensor_2(self):
        """Polls the ADC port and logs the raw voltage data."""
        # Read the ADC value from the sensor adapter (id=1, port=2)
        adc_value = self.ep_sensor_adaptor.get_adc(id=1, port=2)
        # print(f"Sensor adapter id1-port2 ADC is {adc_value}")

        return adc_value

    def adc_to_cm(self, adc_value):
        """
        Converts the ADC reading to distance in centimeters.
        Note: ADC_MAX, M, and C require hardware-specific calibration.
        """
        if adc_value <= 0:
            return -1.0  # Prevent division by zero error

        # Assuming 10-bit ADC resolution (0-1023) and 3.3V maximum system voltage
        ADC_MAX = 1023.0
        SYSTEM_VOLTAGE = 3.3

        # 1. Convert ADC value back to voltage
        voltage = (adc_value / ADC_MAX) * SYSTEM_VOLTAGE

        # Prevent extremely low voltage (sensor reads > 30 cm, outside reliable range)
        if voltage < 0.4:
            return 30.0

        # 2. Calculate distance using the linear equation (Inverse characteristic)
        # Distance L is derived from V_o = M * (1 / (L + 0.42)) + C
        # The M (Slope) and C (Intercept) values below are estimates based on the datasheet
        M = 12.0
        C = 0.0

        distance_cm = (M / (voltage - C)) - 0.42

        # Clamp the value to the 4 to 30 cm range based on sensor specifications
        if distance_cm < 4.0:
            return 4.0
        elif distance_cm > 30.0:
            return 30.0

        return round(distance_cm, 2)

    def _poll_ir_sensors(self):
        """Background thread task to continuously read and log IR sensors."""
        while self.is_logging_ir:
            try:
                # Read raw ADC values
                ir1_raw = self.read_sharp_ir_sensor_1()
                ir2_raw = self.read_sharp_ir_sensor_2()

                # Convert raw values to centimeters
                ir1_cm = self.adc_to_cm(ir1_raw)
                ir2_cm = self.adc_to_cm(ir2_raw)

                # Save both raw and converted values to the single CSV file
                self.save_to_csv(self.ir_file, [ir1_raw, ir1_cm, ir2_raw, ir2_cm])

            except Exception as e:
                print(f"IR Logging Error: {e}")

            # Poll at approximately 10Hz (0.1 seconds) to match other sensor frequencies
            time.sleep(0.1)

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

        # Start background thread for IR sensor polling
        self.is_logging_ir = True
        self.ir_thread = threading.Thread(target=self._poll_ir_sensors)
        self.ir_thread.daemon = True  # Ensure thread stops when main program exits
        self.ir_thread.start()

    def stop_sensors(self):
        """Disable all sensors."""
        time.sleep(self.buffer_time)
        self.ep_chassis.unsub_position()
        self.ep_chassis.unsub_attitude()
        self.ep_chassis.unsub_imu()
        self.ep_chassis.unsub_esc()
        self.ep_sensor.unsub_distance()  # <-- NEW: Stop distance sensor

        # Stop IR logging thread
        self.is_logging_ir = False
        if self.ir_thread is not None:
            self.ir_thread.join()

        print("Data collection and saving to the file have been fully completed.")

    # =========================================================
    # Part 3: Motion Control Functions
    # (Remains identical to your original code)
    # =========================================================
    def move_forward(self, distance=None, speed=None):
        if distance is None:
            distance = self.default_distance
        if speed is None:
            speed = self.default_speed
        print(f"--> Robot moving forward {distance} meters (speed {speed} m/s)")
        self.ep_chassis.move(x=distance, y=0, z=0, xy_speed=speed).wait_for_completed()

    def move_backward(self, distance=None, speed=None):
        if distance is None:
            distance = self.default_distance
        if speed is None:
            speed = self.default_speed
        print(f"--> Robot is moving backward {distance} meters (speed {speed} m/s)")
        self.ep_chassis.move(x=-distance, y=0, z=0, xy_speed=speed).wait_for_completed()

    def turn_left(self, angle=None, speed=None):
        if angle is None:
            angle = self.default_angle
        if speed is None:
            speed = self.default_z_speed
        print(
            f"--> Robot is turning left by {angle} degrees (rotation speed: {speed} deg/s)"
        )
        self.ep_chassis.move(x=0, y=0, z=angle, z_speed=speed).wait_for_completed()

    def turn_right(self, angle=None, speed=None):
        if angle is None:
            angle = self.default_angle
        if speed is None:
            speed = self.default_z_speed
        print(
            f"--> Robot is turning right by {angle} degrees (rotation speed: {speed} deg/s)"
        )
        self.ep_chassis.move(x=0, y=0, z=-angle, z_speed=speed).wait_for_completed()

    def test_movement(self):
        print("--- Starting mobility test ---")
        self.move_forward()
        time.sleep(1)
        self.turn_right()
        self.move_forward()
        time.sleep(1)
        self.turn_right()
        self.move_forward()
        time.sleep(1)
        self.turn_right()
        self.move_forward()
        time.sleep(1)
        self.turn_right()
        print("--- Movement complete ---")
