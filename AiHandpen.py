import cv2
import mediapipe as mp
import numpy as np
import pytesseract
import time
import math

from pathlib import Path
from collections import deque


# ==========================================
# TESSERACT OCR CONFIGURATION
# ==========================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


class HandPen:

    def __init__(self):

        # MediaPipe initialization
        self.mp_hands = mp.tasks.vision

        model_path = Path(__file__).with_name(
            "hand_landmarker.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                "hand_landmarker.task not found!"
            )

        options = self.mp_hands.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=self.mp_hands.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.hands = (
            self.mp_hands.HandLandmarker.create_from_options(
                options
            )
        )

        self.connections = (
            self.mp_hands.HandLandmarksConnections.HAND_CONNECTIONS
        )

        # Drawing
        self.strokes = []
        self.current_stroke = None
        self.previous_point = None

        # Recognition
        self.shape = ""
        self.recognized_text = ""

        self.frame_width = 1280
        self.frame_height = 720

        # Writing state
        self.writing = False
        self.extended_frames = 0
        self.curled_frames = 0

        self.start_frames_required = 2
        self.stop_frames_required = 3

        # Smoothing
        self.point_history = deque(maxlen=3)
        self.smooth_point = None

        # Palm clearing
        self.clear_frames = 0
        self.clear_threshold = 10
        self.clear_latched = False

        # Tracking loss tolerance
        self.missing_frames = 0
        self.max_missing_frames = 2

    # ======================================
    # DISTANCE
    # ======================================

    def distance(self, p1, p2):

        return math.hypot(
            p1.x - p2.x,
            p1.y - p2.y
        )

    # ======================================
    # FINGER ANGLE
    # ======================================

    def finger_angle(self, landmarks, a, b, c):

        p1 = np.array([
            landmarks[a].x,
            landmarks[a].y
        ])

        p2 = np.array([
            landmarks[b].x,
            landmarks[b].y
        ])

        p3 = np.array([
            landmarks[c].x,
            landmarks[c].y
        ])

        v1 = p1 - p2
        v2 = p3 - p2

        denominator = (
            np.linalg.norm(v1) *
            np.linalg.norm(v2)
        )

        if denominator < 1e-8:
            return 0

        cosine = np.dot(v1, v2) / denominator
        cosine = np.clip(cosine, -1.0, 1.0)

        return math.degrees(
            math.acos(cosine)
        )

    # ======================================
    # OPEN PALM DETECTION
    # ======================================

    def is_open_palm(self, landmarks):

        angles = [
            self.finger_angle(landmarks, 5, 6, 8),
            self.finger_angle(landmarks, 9, 10, 12),
            self.finger_angle(landmarks, 13, 14, 16),
            self.finger_angle(landmarks, 17, 18, 20)
        ]

        return all(angle > 150 for angle in angles)

    # ======================================
    # CLOSED FIST DETECTION
    # ======================================

    def is_fist(self, landmarks):

        angles = [
            self.finger_angle(landmarks, 5, 6, 8),
            self.finger_angle(landmarks, 9, 10, 12),
            self.finger_angle(landmarks, 13, 14, 16),
            self.finger_angle(landmarks, 17, 18, 20)
        ]

        return all(angle < 125 for angle in angles)

    # ======================================
    # INDEX FINGER DETECTION
    # ======================================

    def is_index_extended(self, landmarks):

        pip_angle = self.finger_angle(
            landmarks, 5, 6, 8
        )

        dip_angle = self.finger_angle(
            landmarks, 6, 7, 8
        )

        return (
            pip_angle > 150 and
            dip_angle > 145
        )

    # ======================================
    # DRAW HAND LANDMARKS
    # ======================================

    def draw_hand_landmarks(self, frame, landmarks):

        h, w, _ = frame.shape

        for connection in self.connections:

            start = landmarks[connection.start]
            end = landmarks[connection.end]

            x1 = int(start.x * w)
            y1 = int(start.y * h)

            x2 = int(end.x * w)
            y2 = int(end.y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    # ======================================
    # WRITING STATE
    # ======================================

    def update_writing_state(self, landmarks):

        extended = self.is_index_extended(landmarks)

        if extended:

            self.extended_frames += 1
            self.curled_frames = 0

            if self.extended_frames >= self.start_frames_required:
                self.writing = True

        else:

            self.curled_frames += 1
            self.extended_frames = 0

            if self.curled_frames >= self.stop_frames_required:
                self.writing = False

        return self.writing

    # ======================================
    # DRAWING POINT
    # ======================================

    def draw_point(self, frame, landmarks):

        h, w, _ = frame.shape

        index_tip = landmarks[8]

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        self.point_history.append((x, y))

        points = np.array(
            self.point_history,
            dtype=np.float32
        )

        filtered = np.median(
            points,
            axis=0
        )

        target = (
            float(filtered[0]),
            float(filtered[1])
        )

        if self.smooth_point is None:

            self.smooth_point = target

        else:

            alpha = 0.75

            self.smooth_point = (
                alpha * target[0] +
                (1 - alpha) * self.smooth_point[0],

                alpha * target[1] +
                (1 - alpha) * self.smooth_point[1]
            )

        current_point = (
            int(self.smooth_point[0]),
            int(self.smooth_point[1])
        )

        if self.current_stroke is None:

            self.current_stroke = []

            self.strokes.append(
                self.current_stroke
            )

            self.previous_point = current_point

        else:

            if self.previous_point is not None:

                cv2.line(
                    frame,
                    self.previous_point,
                    current_point,
                    (255, 0, 255),
                    7,
                    cv2.LINE_AA
                )

            self.previous_point = current_point

        self.current_stroke.append(current_point)

        cv2.circle(
            frame,
            current_point,
            6,
            (0, 255, 0),
            -1
        )

    # ======================================
    # STOP WRITING
    # ======================================

    def stop_writing(self):

        self.writing = False

        self.extended_frames = 0
        self.curled_frames = 0

        self.current_stroke = None
        self.previous_point = None

        self.point_history.clear()
        self.smooth_point = None

    # ======================================
    # CLEAR SCREEN
    # ======================================

    def clear_screen(self):

        self.strokes.clear()

        self.stop_writing()

        self.shape = ""
        self.recognized_text = ""

        print("Drawing cleared!")

    # ======================================
    # MAIN HAND TRACKING
    # ======================================

    def track_hand(self, frame):

        h, w, _ = frame.shape

        self.frame_width = w
        self.frame_height = h

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = time.monotonic_ns() // 1_000_000

        results = self.hands.detect_for_video(
            mp_image,
            timestamp
        )

        status = "READY"

        if results.hand_landmarks:

            landmarks = results.hand_landmarks[0]

            self.missing_frames = 0

            self.draw_hand_landmarks(
                frame,
                landmarks
            )

            # --------------------------------
            # 1. OPEN PALM = CLEAR
            # --------------------------------

            if self.is_open_palm(landmarks):

                self.clear_frames += 1

                self.stop_writing()

                if (
                    self.clear_frames >= self.clear_threshold
                    and not self.clear_latched
                ):

                    self.clear_screen()
                    self.clear_latched = True

                status = "CLEARING"

            else:

                self.clear_frames = 0
                self.clear_latched = False

                # ----------------------------
                # 2. CLOSED FIST = STOP
                # ----------------------------

                if self.is_fist(landmarks):

                    self.stop_writing()

                    status = "STOPPED"

                else:

                    # ------------------------
                    # 3. INDEX FINGER = WRITE
                    # ------------------------

                    if self.update_writing_state(landmarks):

                        status = "WRITING"

                        self.draw_point(
                            frame,
                            landmarks
                        )

                    else:

                        status = "READY"

                        self.current_stroke = None
                        self.previous_point = None

                        self.point_history.clear()
                        self.smooth_point = None

        else:

            self.missing_frames += 1

            if self.missing_frames > self.max_missing_frames:

                self.stop_writing()

                self.clear_frames = 0
                self.clear_latched = False

            status = "HAND NOT DETECTED"

        # ==================================
        # REDRAW ALL STROKES
        # ==================================

        for stroke in self.strokes:

            for i in range(1, len(stroke)):

                cv2.line(
                    frame,
                    stroke[i - 1],
                    stroke[i],
                    (255, 0, 255),
                    7,
                    cv2.LINE_AA
                )

        # ==================================
        # DISPLAY
        # ==================================

        cv2.rectangle(
            frame,
            (0, 0),
            (w, 115),
            (35, 35, 35),
            -1
        )

        cv2.putText(
            frame,
            f"SHAPE: {self.shape.upper()}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 180, 0),
            2
        )

        cv2.putText(
            frame,
            f"TEXT: {self.recognized_text.upper()[:55]}",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 100),
            2
        )

        status_color = (
            (0, 255, 0)
            if status == "WRITING"
            else (0, 0, 255)
        )

        cv2.putText(
            frame,
            status,
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        return frame

    # ======================================
    # CREATE WHITE CANVAS
    # ======================================

    def create_canvas(self):

        canvas = np.ones(
            (
                self.frame_height,
                self.frame_width,
                3
            ),
            dtype=np.uint8
        ) * 255

        for stroke in self.strokes:

            for i in range(1, len(stroke)):

                cv2.line(
                    canvas,
                    stroke[i - 1],
                    stroke[i],
                    (0, 0, 0),
                    8,
                    cv2.LINE_AA
                )

        return canvas

    # ======================================
    # SHAPE RECOGNITION
    # ======================================

    def recognize_shape(self):

        total_points = sum(
            len(stroke) for stroke in self.strokes
        )

        if total_points < 20:

            print("Not enough points to detect shape!")
            return

        canvas = self.create_canvas()

        gray = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        self.shape = ""

        if not contours:

            print("No shape detected!")
            return

        cnt = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(cnt)

        if area < 2000:

            print("Shape area is too small!")
            return

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        approx = cv2.approxPolyDP(
            cnt,
            0.02 * perimeter,
            True
        )

        sides = len(approx)

        if sides == 3:

            self.shape = "Triangle"

        elif sides == 4:

            x, y, width, height = cv2.boundingRect(
                approx
            )

            ratio = width / float(height)

            if 0.90 <= ratio <= 1.10:

                self.shape = "Square"

            else:

                self.shape = "Rectangle"

        elif sides > 4:

            (x, y), radius = cv2.minEnclosingCircle(
                cnt
            )

            circle_area = np.pi * radius ** 2

            fill_ratio = (
                area / circle_area
                if circle_area > 0
                else 0
            )

            if fill_ratio > 0.75:

                self.shape = "Circle"

            else:

                self.shape = "Polygon"

        print("Recognized Shape:", self.shape)

    # ======================================
    # TEXT RECOGNITION
    # ======================================

    def recognize_text(self):

        total_points = sum(
            len(stroke) for stroke in self.strokes
        )

        if total_points < 20:

            print("Not enough points to recognize text!")
            return

        canvas = self.create_canvas()

        gray = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )

        coords = cv2.findNonZero(
            255 - gray
        )

        if coords is None:

            print("No handwriting found!")
            return

        x, y, width, height = cv2.boundingRect(
            coords
        )

        margin = 25

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)

        x2 = min(
            self.frame_width,
            x + width + margin
        )

        y2 = min(
            self.frame_height,
            y + height + margin
        )

        cropped = gray[y1:y2, x1:x2]

        enlarged = cv2.resize(
            cropped,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        _, processed = cv2.threshold(
            enlarged,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        try:

            self.recognized_text = (
                pytesseract.image_to_string(
                    processed,
                    config="--oem 3 --psm 6"
                ).strip()
            )

            print(
                "Recognized Text:",
                self.recognized_text
            )

        except pytesseract.TesseractNotFoundError:

            print(
                "Tesseract OCR is not installed correctly!"
            )

    # ======================================
    # CLOSE MEDIAPIPE
    # ======================================

    def close(self):

        self.hands.close()


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    hand_pen = HandPen()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        hand_pen.close()

        raise RuntimeError(
            "Cannot open webcam. Check camera permissions."
        )

    print("--------------------------------------")
    print("AI HAND WRITING APPLICATION")
    print("--------------------------------------")
    print("Index finger extended : WRITE")
    print("Closed fist           : STOP")
    print("Open palm             : CLEAR")
    print("S                     : Recognize shape")
    print("T                     : Recognize text")
    print("C                     : Clear drawing")
    print("Q                     : Quit")
    print("--------------------------------------")

    try:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print("Unable to read webcam!")
                break

            frame = cv2.flip(
                frame,
                1
            )

            output = hand_pen.track_hand(frame)

            cv2.imshow(
                "AI Hand Drawing and Recognition",
                output
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):

                hand_pen.recognize_shape()

            elif key == ord('t'):

                hand_pen.recognize_text()

            elif key == ord('c'):

                hand_pen.clear_screen()

            elif key == ord('q'):

                break

    finally:

        cap.release()
        cv2.destroyAllWindows()
        hand_pen.close()


if __name__ == "__main__":

    main()