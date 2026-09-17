import time
from robomaster import blaster

class GimbalController:
    def __init__(self, ep_robot, config):

        self.ep_robot = ep_robot
        self.ep_blaster = ep_robot.blaster

        self.freq_gimbal = config["data_collection"]["frequencies"]["gimbal"]
        self.bullet = getattr(blaster, config["gimbal"]["bullet_type"])


    def gimbal_shoot(self):
        self.ep_blaster.fire(fire_type=self.bullet, times=self.freq_gimbal)