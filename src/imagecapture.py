import cv2
import os
from datetime import datetime

# Create captures folder if it doesn't exist
save_dir = "captures"
os.makedirs(save_dir, exist_ok=True)

#function for adding text boxes to the screen
def draw_text_box(frame, text, x=10, y=10, padding=10, bg_color=(50,50,50), text_color=(255,255,255)):
    # Split text into lines
    lines = text.split("\n")

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Compute width and height based on text
    max_width = 0
    total_height = 0

    # Measure each line
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)
        total_height += h + 5  # small line spacing

    # Rectangle coordinates
    top_left = (x, y)
    bottom_right = (x + max_width + padding * 2, y + total_height + padding)

    # Draw filled rectangle
    cv2.rectangle(frame, top_left, bottom_right, bg_color, cv2.FILLED)

    # Draw text inside
    y_offset = y + padding + 20
    for line in lines:
        cv2.putText(frame, line, (x + padding, y_offset), font, font_scale, text_color, thickness)
        y_offset += h + 5  # move down for next line

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Could not open camera")
    exit()

print("Live preview started")
print("Press ENTER to capture image")
print("Press 'q' to quit")

while True:
    ret, frame = cam.read()
    base_frame = frame.copy()

    if not ret:
        print("Failed to grab frame")
        break
    
    draw_text_box(frame, "press 'ENTER' to capture image or 'q' to quit")
    cv2.imshow("Live Preview", frame)



    key = cv2.waitKey(1)

    # ENTER key = capture
    if key == 13:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{save_dir}/capture_{timestamp}.png"

        cv2.imwrite(filename, base_frame)
        print(f"Saved: {filename}")

    # q = quit
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()