import cv2
import numpy as np


class TargetDetector:

  def __init__(self):
    self.color_ranges = {
        "red": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255])),
        ],
        "yellow": [
            (np.array([20, 160, 100]), np.array([35, 255, 255])),
        ],
        "green": [
            (np.array([40, 70, 70]), np.array([85, 255, 255])),
        ],
        "blue": [
            (np.array([100, 120, 50]), np.array([135, 255, 255])),
        ],
    }

    self.bgr_colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
    }

    # สร้าง Kernel และ CLAHE ล่วงหน้าเพื่อลด Overhead ใน Loop
    self.kernel_open = np.ones((3, 3), np.uint8)
    self.kernel_close = np.ones((7, 7), np.uint8)
    self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

  def _preprocess_hsv(self, frame):
    """ลด Noise เล็กน้อยและเกลี่ยแสงบน Channel V"""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = self.clahe.apply(v)
    return cv2.merge([h, s, v])

  def detect(self, frame, draw=True):
    hsv = self._preprocess_hsv(frame)
    frame_h, frame_w = frame.shape[:2]
    detected_targets = []

    min_area = 800
    max_area = (frame_w * frame_h) * 0.25

    for color_name, ranges in self.color_ranges.items():
      # 1. Masking สี
      mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
      for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)

      # 2. ทำความสะอาด Mask: ลบจุดรบกวน (Open) และเติมรูโหว่จากแสงสะท้อน (Close)
      mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
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

        # 3. กรองวัตถุที่ชนขอบภาพทั้ง 4 ด้าน (แก้ไขบั๊กขอบขวา x + w หายไปในโค้ดเดิม)
        edge_margin = 3
        if (
            x <= edge_margin
            or y <= edge_margin
            or (x + w) >= frame_w - edge_margin
            or (y + h) >= frame_h - edge_margin
        ):
          continue

        # 4. สัดส่วน Aspect Ratio
        aspect_ratio = float(w) / h
        if not (0.7 <= aspect_ratio <= 1.4):
          continue

        # 5. ตรวจสอบรูปทรงให้รัดกุมขึ้น
        shape = "unknown"
        solidity = float(area) / (w * h)

        # ตรวจสอบสี่เหลี่ยม: ต้องมี 4 มุม, เป็น Convex (เส้นไม่เว้าหักมุม), และพื้นที่เต็มกรอบ (Solidity สูง)
        if len(approx) == 4 and cv2.isContourConvex(approx) and solidity > 0.75:
          shape = "rectangle"
        else:
          # ตรวจสอบวงกลม: เทียบพื้นที่ Contour กับ Bounding Circle จริง
          (_, _), radius = cv2.minEnclosingCircle(cnt)
          circle_area = np.pi * (radius**2)
          circularity = (
              4 * np.pi * (area / (peri * peri))
          )  # ยิ่งใกล้ 1.0 ยิ่งกลม

          if circularity > 0.75 and (area / circle_area) > 0.75:
            shape = "circle"

        if shape in ["circle", "rectangle"]:
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