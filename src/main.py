##first test program##

#description
#retrieve data string from file, and use lookup to determine OK or NG condition

# Import necessary packages
import csv
import os
import sys
import cv2
from fps import FPSCounter
from ultralytics import YOLO

# Define path to model and other user variables
model_path = 'models/yolov8s_playing_cards_ncnn_model'  # Path to model
cam_index = 0                          # Index of USB camera
imgW, imgH = 1280, 720                 # Resolution to run USB camera at

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

def getNextBuild(filename='src/buildqueue.txt'):
    """
    Returns the top line of the file and removes it from the source.
    If file is empty, returns None.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    if not lines:
        return None  # empty file

    top_line = lines[0].rstrip("\n")
    remaining = lines[1:]

    with open(filename, "w") as f:
        f.writelines(remaining)

    return top_line

def completeBuild(line, filename="src/processed.txt"):
    """
    Inserts the given line at the top of processed.txt.
    """
    with open(filename, "r") as f:
        existing = f.read()

    with open(filename, "w") as f:
        f.write(line + "\n" + existing)

def previousBuild(processed_file="src/processed.txt", buildqueue_file="src/buildqueue.txt"):
    """
    Reads the top line of processed.txt and copies it to the top of buildqueue.txt
    without removing it from processed.txt.
    """
    # Get top line from processed.txt
    with open(processed_file, "r") as f:
        top_line = f.readline().rstrip("\n")

    if not top_line:
        return None  # processed.txt empty

    # Prepend to buildqueue.txt
    with open(buildqueue_file, "r") as f:
        existing = f.read()

    with open(buildqueue_file, "w") as f:
        f.write(top_line + "\n" + existing)

    return top_line

def lookupCurrent(current, lookup_file="src/playing_cards_lookup.csv"):
    matches = []
    with open(lookup_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            info = [
                row["Description"],
                row["Label Name"],
                row["Quantity Required"],
                row["Message if detected"],
                row["Message if undetected"],
                row["Message Priority"]
            ]

            # CHAR 1 ------------------------------------------------------------
            if row["String Position - Char 1"] != "":
                pos1 = int(row["String Position - Char 1"])

                # SAFETY CHECK for out-of-range
                if pos1 >= len(current):
                    continue  # Cannot match this row

                if current[pos1] != row["Spec Code - Char 1"]:
                    continue
            # If blank → nothing to check (implicit match)

            # CHAR 2 ------------------------------------------------------------
            if row["String Position - Char 2"] != "":
                pos2 = int(row["String Position - Char 2"])

                # SAFETY CHECK for out-of-range
                if pos2 >= len(current):
                    continue

                if current[pos2] != row["Spec Code - Char 2"]:
                    continue

            # CHAR 3 ------------------------------------------------------------
            if row["String Position - Char 3"] != "":
                pos3 = int(row["String Position - Char 3"])

                # SAFETY CHECK for out-of-range
                if pos3 >= len(current):
                    continue

                if current[pos3] != row["Spec Code - Char 3"]:
                    continue

            # If all checks passed
            matches.append(info)

    return matches

def getPartsListPrintable(partsList):
    lines = []

    for part in partsList:
        labelName = part[1]
        qty = int(part[2])

        if qty > 0:
            lines.append(f"{part[0]} ({labelName}) x{qty}")

    return "\n".join(lines)
# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('WARNING: Model path is invalid or model was not found.')
    sys.exit()

# Load the model into memory and get labelmap
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize camera
cap = cv2.VideoCapture(cam_index)
ret = cap.set(3, imgW)
ret = cap.set(4, imgH)

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]


current=getNextBuild()
partsList=lookupCurrent(current)
required=getPartsListPrintable(partsList)

#define fps counter
fps_counter_obj = FPSCounter()

# Begin inference loop
while True:
    
    # Grab frame from counter
    ret, frame = cap.read()
    if (frame is None) or (not ret):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break

    # Run inference on frame with tracking enabled (tracking helps object to be consistently detected in each frame)
    results = model.track(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable to hold every object detected in this frame
    objects_detected = []

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > 0.7:

            # Draw box around object
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            # Draw label for object
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # Add object to list of detected objects
            objects_detected.append(classname)
    output = "OK"
    for part in partsList:
        labelName = part[1]               
        quantityRequired = int(part[2])

        if objects_detected.count(labelName) != quantityRequired:
            output = "NG"
    
    # Draw text box with data
    draw_text_box(frame,(required+"\nOUTPUT: "+ output))
    
    fps_counter_obj.update()
    fps_counter_obj.draw(frame)

    # Display results
    cv2.imshow('Object detection results',frame) # Display image

    # Poll for user keypress and wait 5ms before continuing to next frame
    key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    elif key == ord('m') or key == ord('M'): # Press 'm' to move to the next data string 
        if (output=="OK"):
            completeBuild(current)
            current=getNextBuild()
            partsList=lookupCurrent(current)
            required=getPartsListPrintable(partsList)

# Clean up
cap.release()
cv2.destroyAllWindows()