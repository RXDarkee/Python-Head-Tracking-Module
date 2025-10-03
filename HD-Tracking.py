import cv2
import mediapipe as mp
import numpy as np
import collections
from typing import Tuple, Optional


class HeadTrackingLaser:
    """
    A brutal head-tracking tool that uses a webcam to follow head movements
    and displays a red 'laser' dot at the detected head's center.
    """
    APP_NAME = "Demonic Head Tracking Laser"
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    DOT_COLOR = (0, 0, 255)  # BGR format: Red
    DOT_RADIUS = 15
    SMOOTHING_FACTOR = 0.2   # 0.0 for no smoothing, 1.0 for aggressive following

    def __init__(self):
        # Initialize MediaPipe Face Mesh for robust head detection.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,          # Only interested in tracking one miserable face
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize MediaPipe drawing utilities (unused but available)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Previous head position for smoothing
        self.last_head_pos: Optional[Tuple[int, int]] = None

        self.cap = None  # Webcam capture object

    def _init_webcam(self):
        """Initializes the webcam capture."""
        print("INFO: Attempting to seize your webcam feed...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)

        if not self.cap.isOpened():
            print("FATAL ERROR: Could not open your damn webcam! Is it plugged in?")
            raise IOError("Webcam failed to open. Check hardware and permissions.")
        print(f"INFO: Webcam '{self.cap.getBackendName()}' opened with resolution "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}.")

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """
        Processes a single video frame to detect the head and calculate the laser dot position.
        """
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        head_center_position = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Nose bridge landmark (#6), stable for head center
                target_landmark = face_landmarks.landmark[6]

                x = int(target_landmark.x * self.FRAME_WIDTH)
                y = int(target_landmark.y * self.FRAME_HEIGHT)

                if self.last_head_pos is None:
                    self.last_head_pos = (x, y)
                    head_center_position = (x, y)
                else:
                    smoothed_x = int(self.SMOOTHING_FACTOR * x + (1 - self.SMOOTHING_FACTOR) * self.last_head_pos[0])
                    smoothed_y = int(self.SMOOTHING_FACTOR * y + (1 - self.SMOOTHING_FACTOR) * self.last_head_pos[1])
                    self.last_head_pos = (smoothed_x, smoothed_y)
                    head_center_position = (smoothed_x, smoothed_y)
                break  # Only first detected face

        # Display text feedback
        if head_center_position:
            text = "Face Detected. Laser Locked. No Escape."
            text_color = (0, 255, 0)
        else:
            text = "Searching for your pathetic face..."
            text_color = (0, 0, 255)
            if self.last_head_pos:  # Keep dot at last known position
                head_center_position = self.last_head_pos
            else:
                head_center_position = None

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        return frame, head_center_position

    def run(self):
        """Executes the head tracking and laser display loop."""
        try:
            self._init_webcam()
            print("INFO: Tracking initiated. Press 'q' to brutally terminate the tracking.")

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("WARNING: Failed to grab a frame from the webcam.")
                    break

                processed_frame, dot_position = self._process_frame(frame)

                if dot_position:
                    cv2.circle(processed_frame, dot_position, self.DOT_RADIUS, self.DOT_COLOR, -1)

                cv2.imshow(self.APP_NAME, processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("INFO: Termination signal 'q' received.")
                    break

        except Exception as e:
            print(f"CRITICAL SYSTEM FAILURE: {e}")
        finally:
            self._release_resources()

    def _release_resources(self):
        """Releases the webcam and destroys all OpenCV windows."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("INFO: Webcam released.")
        cv2.destroyAllWindows()
        print("INFO: All display windows destroyed.")


if __name__ == "__main__":
    laser_tracker = HeadTrackingLaser()
    laser_tracker.run()
