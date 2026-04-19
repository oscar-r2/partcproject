#Title: "Optimising Object Detection Systems on Low-Cost Edge Devices for Automotive Manufacturing - Deployment Prototype"
#Purpose: Demonstrate how a YOLO model can be used to verify a physical assembly process
#Limitations:
#   -The ability to alter confidence threshold class-by-class has not been added
#   -The program works with a USB webcam OR Pi Camera Module 3 by commenting out certain lines - this cannot be changed
#    without source code access
#   -No logging or frame saving is present
#   -All users can perform a manual override, no administrator login is required
#   -The user has no visual indication that individual class quantities are satisfied, only bounding boxes and overall
#    condition (OK/NG)
#Acknowledgements
#   This program is adapted from "candy_calorie_counter.py" by EJ Electronics Constultants, available at
#   https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/examples/candy_calorie_counter/candy_calorie_counter.py


# Imports
import csv
import os
import sys
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import controls
import warnings
from gpiozero import LED

# Define path to model and other user variables
model_path = 'models/model18_RPI4B_070426_ncnn_model'  # Path to model
cam_index = 0                          # Index of USB camera
imgW, imgH = 1280, 720                 # Resolution to run USB camera at
imgsz = 512                             # Resolution to run model at
led= LED(23)
light_status = False

#Function to create a text box with given text at a specified location
def draw_text_box(frame, text, corner="top-left", padding=10,
                  margin=10,
                  bg_color=(50, 50, 50),
                  text_color=(255, 255, 255)):

    lines = text.split("\n")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_spacing = 5

    max_width = 0
    total_height = 0
    line_sizes = []

    # Measure each line
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)
        total_height += h + line_spacing
        line_sizes.append((w, h))

    total_height -= line_spacing  # remove extra spacing from last line

    # Frame dimensions
    frame_h, frame_w = frame.shape[:2]

    box_width = max_width + padding * 2
    box_height = total_height + padding * 2

    # Determine position based on corner
    if corner == "top-left":
        x = margin
        y = margin
    elif corner == "top-right":
        x = frame_w - box_width - margin
        y = margin
    elif corner == "bottom-left":
        x = margin
        y = frame_h - box_height - margin
    elif corner == "bottom-right":
        x = frame_w - box_width - margin
        y = frame_h - box_height - margin
    else:
        raise ValueError("corner must be one of: "
                         "'top-left', 'top-right', "
                         "'bottom-left', 'bottom-right'")

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x, y),
        (x + box_width, y + box_height),
        bg_color,
        cv2.FILLED
    )

    # Draw text
    y_offset = y + padding
    for (line, (_, h)) in zip(lines, line_sizes):
        y_offset += h
        cv2.putText(
            frame,
            line,
            (x + padding, y_offset),
            font,
            font_scale,
            text_color,
            thickness
        )
        y_offset += line_spacing

#Function to return the top line of the build queue file without removing it from the source 
def getNextBuild(filename='src/buildqueue.txt'):
    
    with open(filename, "r") as f:
        lines = f.readlines()

    if not lines:
        return None  # empty file

    top_line = lines[0].rstrip("\n")

    return top_line

#Inserts the given line at the top of processed.txt, and removes the top line of buildqueue.txt
def completeBuild(line, processedfilename="src/processed.txt", buildqueuefilename="src/buildqueue.txt"):
    with open(processedfilename, "r") as f:
        existing = f.read()

    with open(processedfilename, "w") as f:
        f.write(line + "\n" + existing)

    with open(buildqueuefilename, "r") as f:
        remaining = f.readlines()[1:]

    with open(buildqueuefilename, "w") as f:
        f.writelines(remaining)

#Moves the top line of processed.txt to the top of buildqueue.txt and removes it from processed.txt.
def previousBuild(processed_file="src/processed.txt", buildqueue_file="src/buildqueue.txt"):

    # Read all lines from processed.txt
    with open(processed_file, "r") as f:
        lines = f.readlines()

    if not lines:
        return None  # processed.txt empty

    top_line = lines[0].rstrip("\n")
    remaining_lines = lines[1:]

    # Rewrite processed.txt without the top line
    with open(processed_file, "w") as f:
        f.writelines(remaining_lines)

    # Read existing buildqueue.txt content
    with open(buildqueue_file, "r") as f:
        existing = f.read()

    # Prepend the removed line to buildqueue.txt
    with open(buildqueue_file, "w") as f:
        f.write(top_line + "\n" + existing)

    return top_line

#Function to use the lookup table to turn the data string into a parts list
def lookupCurrent(current, lookup_file="src/lookup_v2.csv"):
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

#Return a string of the parts list to display onscreen for the user
def getPartsListPrintable(partsList):
    
    lines = []

    for part in partsList:
        labelName = part[1]
        qty = int(part[2])

        if qty > 0:
            lines.append(f"{part[0]} ({labelName}) x{qty}")

    return "\n".join(lines)

#A class to measure the current FPS and display it on screen at all times
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
        draw_text_box(frame,"FPS: "+ str((round(self.fps,2))),"top-right")
        return frame

# A class to display a list of commands on screen for the user   
class Menu:
    def __init__(self):
        self.capture_mode = 0 
        self.capture_modes = [[0,"cam only"],[1,"bounding boxes"],[2, "full screenshot"]]
        self.options = "q-quit\ns-pause\nc-capture image - "+self.capture_modes[self.capture_mode][1]+"\nm-next capture mode\nn-next data\nf-force next\nt-toggle inference"
        
    def update_capture_mode(self):
        if self.capture_mode==2:
            self.capture_mode =0
        else:
            self.capture_mode +=1
        self.options = "q-quit\ns-pause\nc-capture image - "+self.capture_modes[self.capture_mode][1]+"\nm-next capture mode\nn-next data\nt-toggle inference"

    def get_capture_mode(self):
        return self.capture_mode
    
    def get_capture_mode_printable(self):
        return self.capture_modes[self.capture_mode,1]


    def draw(self, frame):
        draw_text_box(frame,self.options,"bottom-left")

        return frame

# A class to take screenshots with or without certain UI elements    
class CaptureSaver:
    def _init_(self):
        save_dir=""

    def save(self,base_frame,bounding_box_frame,frame,capture_mode):
        filename = timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if capture_mode == 0:
            save_dir = "captures/camera_only_captures"
            write_frame = base_frame  
        elif capture_mode == 1:
            save_dir = "captures/bouding_box_captures"  
            write_frame = bounding_box_frame   
        elif capture_mode == 2:
            save_dir = "captures/full screenshots"  
            write_frame = frame   
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/capture_{timestamp}.png"
        cv2.imwrite(filename, write_frame)
        print(f"Saved: {filename}")
        
# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('WARNING: Model path is invalid or model was not found.')
    sys.exit()

# Load the model into memory and get labelmap
model = YOLO(model_path, task='detect')
labels = model.names

# Initialise camera - USB WEBCAM ONLY - COMMENT OUT IF USING PICAM
#cap = cv2.VideoCapture(cam_index)
#ret = cap.set(3, imgW)
#ret = cap.set(4, imgH)
#end of webcam settings

# Initialise camera - PICAM MODEL 3 ONLY - COMMENT OUT IF USING WEBCAM
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()
#Set autofocus to continuous mode
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
#end of picam settings

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

#get parts list based off next available data string
current=getNextBuild()
partsList=lookupCurrent(current)
required=getPartsListPrintable(partsList)

#define fps counter and menu objects
fps_counter_obj = FPSCounter()
menu_obj = Menu()
capture_obj = CaptureSaver()
inference_enabled = 1
menu=True

# Begin inference loop
while True:
    # Grab frame from counter
    frame = picam2.capture_array() #picam - comment out if using webcam
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #picam - comment out if using webcam
    #ret, frame = cap.read() #webcam - comment out if using picam
    if (frame is None): #or (not ret):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break

    # save base frame before inference, in case user wants to record it later
    base_frame = frame.copy()
    bounding_box_frame = frame.copy()

    if inference_enabled:    
        # Run inference on frame with tracking enabled (tracking helps object to be consistently detected in each frame)
        results = model.track(frame, verbose=False, imgsz=imgsz)

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
            if conf > 0.3:

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

            bounding_box_frame=frame.copy()

        output = "OK"
        for part in partsList:
            labelName = part[1]               
            quantityRequired = int(part[2])

            if objects_detected.count(labelName) != quantityRequired:
                output = "NG"
        
        # Draw text box with data
        draw_text_box(frame,(required+"\nOUTPUT: "+ output))
    
    else:
        draw_text_box(frame,"Inference disabled.\nPress 't' to toggle")

    fps_counter_obj.update()
    fps_counter_obj.draw(frame)
    if menu==True:
        menu_obj.draw(frame)

    # Display results
    cv2.imshow('Object Detection System',frame) # Display image

    # Poll for user keypress and wait 1ms before continuing to next frame
    key = cv2.waitKey(1)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('t') or key == ord('T'): # Press 't' to toggle inference
        inference_enabled=not (inference_enabled)
    elif key == ord('c') or key == ord('C'): # Press 'c' to save a picture of results on this frame
        capture_obj.save(base_frame, bounding_box_frame, frame, menu_obj.get_capture_mode())
    elif key == ord('m') or key == ord('M'): # Press 'm' to change capture mode
        menu_obj.update_capture_mode()
    elif key == ord('n') or key == ord('N'): # Press 'n' to move to the next data string 
        if (output=="OK"):
            completeBuild(current)
            current=getNextBuild()
            partsList=lookupCurrent(current)
            required=getPartsListPrintable(partsList)
    elif key == ord('f') or key == ord('F'): # Press 'f' to force next build
            completeBuild(current)
            current=getNextBuild()
            partsList=lookupCurrent(current)
            required=getPartsListPrintable(partsList)
    elif key == ord('p') or key == ord('P'): # Press 'p' to return to previous build
        current=previousBuild()
        partsList=lookupCurrent(current)
        required=getPartsListPrintable(partsList)
    elif key == ord('l') or key == ord('L'): # Press 'l' to toggle light
        light_status = not light_status
        if (light_status == True):
            led.on()
        elif (light_status == False):
            led.off()
    elif key == ord('h') or key == ord('H'): # Press 'h' to toggle menu
        menu=not(menu)
        
# Clean up
cv2.destroyAllWindows()
