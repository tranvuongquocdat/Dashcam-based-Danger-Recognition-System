import cv2
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from ultralytics import YOLO
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
from utils import *


# load the pre-trained YOLOv8n model
coco_model = YOLO(r"model\yolov8n.pt")
#load traffic_sign model
traffic_sign_model = YOLO(r"model\traffic_sign_recognition.pt")
#load pohole model
pohole_model = YOLO(r"model\pohole.pt")

init_time = time.time()
last_played = {}
for value in ['person', 'bicycle', 'car']:
    last_played[value] = init_time

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

class App:
    def __init__(self, window, window_title = 'ADSV', video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # Define the COCO classes
        self.SCREEN_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'traffic light', 'stop sign']
        self.SOUND_CLASSES = ['person', 'bicycle', 'car']
        
        # Create an entry for the video source
        self.entry = tk.Entry(window, width=30)
        self.entry.pack()
        self.entry.insert(0, video_source)

        # Create checkboxes for screen and sound
        self.screen_vars = [tk.BooleanVar(value = True) for _ in self.SCREEN_CLASSES]
        self.sound_vars = [tk.BooleanVar(value = True) for _ in self.SOUND_CLASSES]

        # Create options menu for screen and sound
        self.screen_button = tk.Button(window, text="Screen", command=self.create_screen_window)
        self.screen_button.pack()
        
        self.sound_button = tk.Button(window, text="Sound", command=self.create_sound_window)
        self.sound_button.pack()

        self.vid = cv2.VideoCapture(self.video_source)

        # Set up canvas for video
        self.canvas = tk.Canvas(window, width = 1080, height = 720) 
        self.canvas.pack()

        self.btn_start= tk.Button(window, text="Start", width=30, command=self.start_video)
        self.btn_start.pack(anchor=tk.CENTER, expand=True)

        self.btn_pause= tk.Button(window, text="Pause/Play", width=30, command=self.pause_play_video)
        self.btn_pause.pack(anchor=tk.CENTER, expand=True)

        # Set up labels for FPS
        self.fps_label = tk.Label(window)
        self.fps_label.pack()
        self.last_frame_time = 0

        self.delay = 1  # Increase FPS by reducing delay
        self.update()

        self.window.mainloop()

    def start_video(self):
        self.video_source = self.entry.get()  # Get the video source from the entry
        self.vid = cv2.VideoCapture(self.video_source)
        self.update()

    def pause_play_video(self):
        self.delay = 0 if self.delay == 1 else 1
        self.update()

    def update(self):
        # Get the FPS of the original video
        fps = self.vid.get(cv2.CAP_PROP_FPS)
        # If the FPS is zero, we can manually set it to a sensible value
        if fps == 0:
            # Assume a standard video fps
            fps = 30 
        frame_delay = 1/fps  # delay for each frame in seconds

        frame_count = 0  # Initialize frame count
        skip_frames = 0  # Number of frames to skip
        processed_frames = 1  # Keep track of processed frames
        start_time = time.time()  # Start time of processing
        screen_values = [cls for cls, var in zip(self.SCREEN_CLASSES, self.screen_vars) if var.get()]
        sound_values = [cls for cls, var in zip(self.SOUND_CLASSES, self.sound_vars) if var.get()]

        ret, frame = self.vid.read()
        if ret==True:
            frame_count += 1
            if frame_count % (skip_frames + 1) == 0:  # Process frame if frame_count is divisible by (skip_frames + 1)
                processed_frames += 1

                # process your frame here
                #frame = cv2.resize(frame, (1080, 720))

                coco_result = coco_model(frame)[0].boxes.data.tolist()
                coco_result_screen = [res for res in coco_result if coco_label_map[str(int(res[5]))] in screen_values]
                traffic_sign_result = traffic_sign_model(frame)[0].boxes.data.tolist()
                traffic_sign_result_screen = [res for res in traffic_sign_result if traffic_sign_label_map[str(int(res[5]))] in screen_values]
                pohole_result = pohole_model(frame)[0].boxes.data.tolist()
                pohole_result_screen = [res for res in pohole_result if pohole_label_map[str(int(res[5]))] in screen_values]

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = forward(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
                #draw the result
                frame = draw_detections(frame, coco_result_screen, tag = 'coco')
                frame = draw_detections(frame, traffic_sign_result_screen, tag = 'traffic_sign')
                frame = draw_detections(frame, pohole_result_screen, tag = 'pohole')

                #get object for playsound
                # Create a list of classes detected in this frame
                frame_classes = [coco_label_map[str(int(res[5]))] for res in coco_result]
                frame_classes += [traffic_sign_label_map[str(int(res[5]))] for res in traffic_sign_result]
                frame_classes += [pohole_label_map[str(int(res[5]))] for res in pohole_result]

                #play_sound_alerts
                for cls in set(frame_classes):
                    if cls in sound_values:
                        # If this class it was played more than n seconds ago
                        print('detected: ' + cls)
                        print(time.time() - last_played[cls])
                        if time.time() - last_played[cls] >= 10:
                            play_alert_sound(cls)
                            last_played[cls] = time.time() 


                # Put FPS on the top-left corner of the frame
                fps = processed_frames / (time.time() - start_time)
                cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.fps_label.config(text='FPS: %.2f' % fps)

                # adjust wait time based on the original FPS
                while time.time() - start_time < frame_delay:
                    pass

        if self.delay != 0:
            self.window.after(self.delay, self.update)

    def create_screen_window(self):
        screen_window = tk.Toplevel(self.window)
        screen_window.title("Screen Options")
        self.screen_checkboxes = [tk.Checkbutton(screen_window, text=cls, var=var) for cls, var in zip(self.SCREEN_CLASSES, self.screen_vars)]
        for cb in self.screen_checkboxes:
            cb.pack(anchor=tk.W)

    def create_sound_window(self):
        sound_window = tk.Toplevel(self.window)
        sound_window.title("Sound Options")
        self.sound_checkboxes = [tk.Checkbutton(sound_window, text=cls, var=var) for cls, var in zip(self.SOUND_CLASSES, self.sound_vars)]
        for cb in self.sound_checkboxes:
            cb.pack(anchor=tk.W)

if __name__ == "__main__":
    App(tk.Tk(), "Risk Alert System")
