import numpy as np
import cv2
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
import cv2
import numpy as np
import os
import threading
from playsound import playsound
import os
import threading

calibration = CameraCalibration('camera_cal', 9, 6)
thresholding = Thresholding()
transform = PerspectiveTransformation()
lanelines = LaneLines()

coco_label_map = {'0': 'human', 
                     '1': '', 
                     '2': 'car', 
                     '3': '', 
                     '4': 'motorcycle', 
                     '5': '', 
                     '6': 'bus', 
                     '7': '', 
                     '8': 'truck', 
                     '9': 'traffic_light', 
                     '10': '', 
                     '11': '', 
                     '12': '', 
                     '13': '', 
                     '14': '', 
                     '15': '', '16': '', '17': '', '18': '', '19': '', '20': '', '21': '', '22': '', '23': '', '24': '', '25': '', '26': '', '27': '', '28': '', '29': '', '30': '', '31': '', '32': '', '33': '', '34': '', '35': '', '36': '', '37': '', '38': '', '39': '', '40': '', '41': '', '42': '', '43': '', '44': '', '45': '', '46': '', '47': '', '48': '', '49': '', '50': '', '51': '', '52': '', '53': '', '54': '', '55': '', '56': '', '57': '', '58': '', '59': '', '60': '', '61': '', '62': '', '63': '', '64': '', '65': '', '66': '', '67': '', '68': '', '69': '', '70': '', '71': '', '72': '', '73': '', '74': '', '75': '', '76': '', '77': '', '78': '', '79': ''}

traffic_sign_label_map = {'0': 'trafficlight',
                     '1': 'speedlimit',
                     '2': 'crosswalk',
                     '3': 'stop'
                }
pohole_label_map = {'0': 'pohole'}

def forward(img):
    out_img = np.copy(img)
    img = calibration.undistort(img)
    img = transform.forward(img)
    img = thresholding.forward(img)
    img = lanelines.forward(img)
    img = transform.backward(img)

    img = cv2.resize(img, (out_img.shape[1], out_img.shape[0]))

    out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
    out_img = lanelines.plot(out_img)
    return out_img

def draw_detections(frame, last_detections, tag = 'coco'):
    if (tag == 'coco'):
        COLOR = (153, 255, 204)
        label_map = coco_label_map
    elif (tag == 'traffic_sign'):
        COLOR = (204, 133, 255)
        label_map = traffic_sign_label_map
    elif tag == 'pohole':
        COLOR = (255, 133, 204)
        label_map =pohole_label_map 

    # define some constants
    CONFIDENCE_THRESHOLD = 0.2
    if last_detections is not None:
        for data in last_detections:
            # extract the confidence (i.e., probability) associated with the detection
            confidence = data[4]

            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence,
            # draw the bounding box on the frame
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            corner_length = 7  # Ví dụ, độ dài của mỗi góc là 20 pixel

            # Vẽ 4 góc của hình chữ nhật
            cv2.line(frame, (xmin, ymin), (xmin + corner_length, ymin), COLOR, 1)
            cv2.line(frame, (xmin, ymin), (xmin, ymin + corner_length), COLOR, 1)
            cv2.line(frame, (xmax, ymin), (xmax - corner_length, ymin), COLOR, 1)
            cv2.line(frame, (xmax, ymin), (xmax, ymin + corner_length), COLOR, 1)
            cv2.line(frame, (xmin, ymax), (xmin + corner_length, ymax), COLOR, 1)
            cv2.line(frame, (xmin, ymax), (xmin, ymax - corner_length), COLOR, 1)
            cv2.line(frame, (xmax, ymax), (xmax - corner_length, ymax), COLOR, 1)
            cv2.line(frame, (xmax, ymax), (xmax, ymax - corner_length), COLOR, 1)

            # Get the class ID and draw it on the frame
            class_id = data[5]
            text = f"{label_map[str(int(class_id))]},{confidence:.2f}"
            cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)

    return frame

def play_alert_sound(cls):
    # Create the file name based on the class
    file_name = f"sound_alert/{cls}_alert.mp3"
    print('sound alert ' + cls)
    # Check if the file exists
    if os.path.exists(file_name):
        # Play the audio file in a new thread
        threading.Thread(target=playsound, args=(file_name,)).start()