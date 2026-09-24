import cv2
import numpy as np


class TargetDetector:

    def __init__(self):
        self.color_ranges = {
            "red": [
                (np.array([0, 100, 70]), np.array([12, 255, 255])),
                (np.array([168, 100, 70]), np.array([180, 255, 255])),
            ],
            "yellow": [
                (np.array([20, 120, 70]), np.array([38, 255, 255])),
            ],
            "green": [
                (np.array([30, 35, 20]), np.array([95, 255, 255])),
            ],
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
        self.kernel_close = np.ones((5, 5), np.uint8)
        self.kernel_edge = np.ones((3, 3), np.uint8)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess_hsv(self, frame):
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = self.clahe.apply(v)
        return cv2.merge([h, s, v])

    def _sobel_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gradient_x, gradient_y)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(magnitude, 45, 255, cv2.THRESH_BINARY)
        return cv2.dilate(edges, self.kernel_edge, iterations=1)

    @staticmethod
    def _angle_cosine(point_a, point_b, point_c):
        vector_a = point_a - point_b
        vector_c = point_c - point_b
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_c)
        if denominator == 0:
            return 1.0
        return abs(float(np.dot(vector_a, vector_c) / denominator))

    def _classify_shape(self, contour, area, width, height):
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "unknown"

        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        vertices = len(polygon)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        _, enclosing_radius = cv2.minEnclosingCircle(contour)
        enclosing_area = np.pi * enclosing_radius ** 2
        circle_fill = area / enclosing_area if enclosing_area else 0.0

        if vertices == 3:
            return "triangle"

        if vertices == 4:
            points = polygon.reshape(4, 2).astype(np.float32)
            cosines = [
                self._angle_cosine(
                    points[(index - 1) % 4],
                    points[index],
                    points[(index + 1) % 4],
                )
                for index in range(4)
            ]
            rectangularity = area / float(width * height)
            if max(cosines) < 0.35 and rectangularity > 0.65:
                return "rectangle"

        if circularity > 0.72 and circle_fill > 0.72:
            return "circle"

        if vertices >= 5 and circularity > 0.60 and circle_fill > 0.60:
            return "circle"

        return "unknown"

    def detect(self, frame, draw=True):
        hsv = self._preprocess_hsv(frame)
        sobel_edges = self._sobel_edges(frame)
        frame_h, frame_w = frame.shape[:2]
        detected_targets = []

        min_area = 400
        max_area = (frame_w * frame_h) * 0.20

        for color_name, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            # Use edges only next to detected color, so background edges do not become targets.
            nearby_color = cv2.dilate(mask, self.kernel_edge, iterations=1)
            edge_hint = cv2.bitwise_and(sobel_edges, nearby_color)
            mask = cv2.bitwise_or(mask, edge_hint)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=2)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # กรองพื้นที่เหนือพื้นห้อง
                if (y + h) < (frame_h * 0.35):
                    continue

                # กรองขอบจอภาพ
                edge_margin = 3
                if (
                    x <= edge_margin
                    or y <= edge_margin
                    or (x + w) >= frame_w - edge_margin
                    or (y + h) >= frame_h - edge_margin
                ):
                    continue

                # กรอง Aspect Ratio
                aspect_ratio = float(w) / h
                if not (0.50 <= aspect_ratio <= 1.50):
                    continue

                # กรอง Solidity
                solidity = float(area) / (w * h)
                if solidity < 0.40:
                    continue

                shape = self._classify_shape(cnt, area, w, h)

                if shape == "unknown":
                    continue

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