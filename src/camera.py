import time
import cv2
import os

class CameraController:
    def __init__(self, ep_robot, config):
         self.ep_robot = ep_robot
         self.ep_camera = ep_robot.camera

         self.data_dir = config["data_collection"]["data_dir"]
         self.default_duration = config["data_collection"]["frequencies"]["camera"]

         os.makedirs(self.data_dir, exist_ok=True)

    def start_camera(self):
         self.ep_camera.start_video_stream(display=False)
         time.sleep(0.5)

    def capture(self,duration = None):
        if duration is None:
            duration = self.default_duration

        for i in range(duration):
            img = self.ep_camera.read_cv2_image(strategy="newest")

            if img is None:
                time.sleep(0.1)
                continue

            # Generate unique filename with timestamp and index
            timestamp = int(time.time() * 1000)
            filename = f"img_{i:04d}_{timestamp}.jpg"
            filepath = os.path.join(self.data_dir, filename)

            # Save frame to disk
            cv2.imwrite(filepath, img)

            # Display frame
            cv2.imshow("Robot Camera", img)

            # Break early if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(1)

    def stop_camera(self):
        cv2.destroyAllWindows()
        self.ep_camera.stop_video_stream()