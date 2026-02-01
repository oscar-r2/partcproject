import time
import cv2

class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def update(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return self.fps
    
    def draw(self, frame):
        h, w = frame.shape[:2]

        text = f"FPS: {int(self.fps)}"

        # Measure text size so it sits neatly in the corner
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            2
        )

        x = w - text_width - 10
        y = text_height + 10

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return frame