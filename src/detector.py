import cv2
import numpy as np


class TargetDetector:

    def __init__(self):
        self.color_ranges = {
            # แดง: ตัดเงาและคราบส้ม
            "red": [
                (np.array([0, 100, 70]), np.array([12, 255, 255])),
                (np.array([168, 100, 70]), np.array([180, 255, 255])),
            ],
            # เหลือง: บังคับ S >= 145 ตัดผนังห้องสีครีมโดยเด็ดขาด
            "yellow": [
                (np.array([20, 120, 70]), np.array([38, 255, 255])),
            ],
            # เขียว: เพิ่ม S >= 60 และ V >= 50 เพื่อไม่ให้จับมุมมืด/เงาขอบประตู
            "green": [
                (np.array([40, 60, 35]), np.array([80, 255, 255])),
            ],
            # น้ำเงิน: เพิ่ม S >= 90 เพื่อไม่ให้จับเงาสะท้อนกระเบื้องจางๆ
            "blue": [
                (np.array([95, 70, 35]), np.array([135, 255, 255])),
            ],
        }

        self.bgr_colors = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
        }

        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((7, 7), np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess_hsv(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = self.clahe.apply(v)
        return cv2.merge([h, s, v])

    def detect(self, frame, draw=True):
        hsv = self._preprocess_hsv(frame)
        frame_h, frame_w = frame.shape[:2]
        detected_targets = []

        min_area = 400
        max_area = (frame_w * frame_h) * 0.20

        for color_name, ranges in self.color_ranges.items():
            # 1. สร้าง Mask
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            # 2. ทำความสะอาด Mask (เพิ่ม iterations=2 เพื่อลบเม็ด Noise ฝุ่น/เงาของสีเขียวให้หมด)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue

                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)

                # 3. ตัดพื้นที่เหนือพื้นห้อง (เพดาน, ประตู, สวิตช์ไฟ)
                if (y + h) < (frame_h * 0.35):
                    continue

                # 4. กรองขอบจอภาพ
                edge_margin = 3
                if (
                    x <= edge_margin
                    or y <= edge_margin
                    or (x + w) >= frame_w - edge_margin
                    or (y + h) >= frame_h - edge_margin
                ):
                    continue

                # 5. สัดส่วน Aspect Ratio (ตัดแสงสะท้อนแนวยาวบนกระเบื้อง)
                aspect_ratio = float(w) / h
                if not (0.50 <= aspect_ratio <= 1.50):
                    continue

                # 6. กรองกำแพง/แสงเงาเว้าแหว่งด้วย Solidity
                solidity = float(area) / (w * h)
                if solidity < 0.65:
                    continue

                # ==============================================================
                # 7. แยกวงกลม vs สี่เหลี่ยม ด้วย Enclosing Circle Ratio (แม่นยำสูง)
                # ==============================================================
                (_, _), radius = cv2.minEnclosingCircle(cnt)
                circle_area = np.pi * (radius ** 2)
                circle_ratio = area / circle_area if circle_area > 0 else 0

                # - วงกลมจริง: จะกินพื้นที่วงกลมล้อมรอบ > 76% (circle_ratio >= 0.76)
                # - สี่เหลี่ยมจัตุรัส: ตามทฤษฎีเรขาคณิตจะกินพื้นที่เพียง ~63.6% ของวงกลมล้อมรอบ
                if circle_ratio >= 0.76:
                    shape = "circle"
                else:
                    shape = "rectangle"

                center = (x + w // 2, y + h // 2)
                target_info = {
                    "shape": shape,
                    "color": color_name,
                    "center": center,
                    "bbox": (x, y, w, h),
                }
                detected_targets.append(target_info)

                if draw:
                    box_color = self.bgr_colors.get(color_name, (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.circle(frame, center, 4, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"{color_name} {shape}",
                        (x, max(15, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )

        return frame, detected_targets