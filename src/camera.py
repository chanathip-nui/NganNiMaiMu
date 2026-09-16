import time
import cv2

class CameraController:
    def __init__(self, ep_robot, config):
         self.ep_robot = ep_robot
         self.ep_camera = ep_robot.camera

         self.default_duration = config["data_collection"]["frequencies"]["camera"]

    def start_camera(self):
         self.ep_camera.start_video_stream(display=False)
         time.sleep(0.5)

    def capture(self,duration = None):
        duration = self.default_duration
        for i in range(0, duration):
            img = self.ep_camera.read_cv2_image(strategy="newest")
            cv2.imshow("Robot", img)
            cv2.waitKey(1)
            time.sleep(1)

    def stop_camera(self):
        cv2.destroyAllWindows()
        self.ep_camera.stop_video_stream()