import time
from robomaster import blaster

class PIDController:
    def __init__(self, kp, ki, kd, limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_limit, self.max_limit = limits
        
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.filtered_d = 0.0  #  Derivative Noise

    def compute(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            dt = 0.01

        # 1. Proportional term
        p_term = self.kp * error

        # 2. Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # 3. Derivative term + LPF
        derivative = (error - self.last_error) / dt
        self.filtered_d = 0.7 * self.filtered_d + 0.3 * derivative #Low-Pass Filter
        d_term = self.kd * self.filtered_d

        output = p_term + i_term + d_term
        output = max(self.min_limit, min(self.max_limit, output))

        self.last_error = error
        self.last_time = now
        return output

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.filtered_d = 0.0
        self.last_time = time.time()

class GimbalController:
    def __init__(self, ep_robot, config):

        self.ep_robot = ep_robot
        self.ep_gimbal = ep_robot.gimbal
        self.ep_blaster = ep_robot.blaster

        self.freq_gimbal = config["data_collection"]["frequencies"]["gimbal"]
        self.bullet = getattr(blaster, config["gimbal"]["bullet_type"])
        self.target_marker = str(config["gimbal"]["target_marker"])
        self.threshold =config["gimbal"]["error_threshold"]

        yaw_pid = config["gimbal"]["pid_yaw"]
        pitch_pid = config["gimbal"]["pid_pitch"]

        self.pid_yaw = PIDController(
            kp=yaw_pid["kp"],
            ki=yaw_pid["ki"],
            kd=yaw_pid["kd"],
            limits=(-yaw_pid["limit"], yaw_pid["limit"])
        )
        self.pid_pitch = PIDController(
            kp=pitch_pid["kp"],
            ki=pitch_pid["ki"],
            kd=pitch_pid["kd"],
            limits=(-pitch_pid["limit"], pitch_pid["limit"])
        )

        self.target_locked = False

    def gimbal_shoot(self):
        self.ep_blaster.fire(fire_type=self.bullet, times=self.freq_gimbal)

    def reset_gimbal(self):
        """หยุดมอเตอร์ สั่งกิมบอลกลับมาจุดศูนย์กลาง (0, 0) และรีเซ็ตค่า PID ทั้งหมด"""
        self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        self.ep_gimbal.recenter().wait_for_completed()
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.target_locked = False
        print(">> Gimbal recentered and PID reset.")

    def aim_and_shoot(self, target_center, frame_width, frame_height):
        """คำนวณ Error เล็ง และยิง คืนค่า True เมื่อยิงสำเร็จ"""
        cx, cy = target_center
        err_x = (cx / frame_width) - 0.5
        err_y = 0.5 - (cy / frame_height)

        yaw_speed = self.pid_yaw.compute(err_x)
        pitch_speed = self.pid_pitch.compute(err_y)

        # เมื่อเล็งเข้าเป้าตาม Deadband
        if abs(err_x) < self.threshold and abs(err_y) < self.threshold:
            self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            print(">> Locked on Target! Firing...")
            self.gimbal_shoot()
            self.pid_yaw.reset()
            self.pid_pitch.reset()
            return True  # ยิงสำเร็จ
        else:
            self.ep_gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)
            return False