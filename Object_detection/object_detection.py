import cv2
import tensorflow as tf
from object_detection_functions import *
import threading
import torch
import serial
import time
from FaceModel import FaceModel

path = "C:\\Users\\orucc\\Desktop\\oruc_and_erdem"

face_model= FaceModel(path)

# load the model that will be used
# get the pretrained model

#module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
#detector = hub.load(module_handle).signatures['default']

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 'yolov5s' is the smallest model. You can also use 'yolov5m', 'yolov5l', or 'yolov5x'.

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize lock and frame variable
frame_lock = threading.Lock()
shared_frame = None

def display_video():
    global shared_frame
    # Capture video from the specified source
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with frame_lock:
            shared_frame = frame.copy()

        # Display the frame
        cv2.imshow('Video Frame', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows when done
    cap.release()
    cv2.destroyAllWindows()


# Create a thread for the video display function with the video source argument
video_thread = threading.Thread(target=display_video)

video_thread.start()

frame_for_model = None


arduinoPort = "com6"
arduinoData = serial.Serial(arduinoPort,"9600")
time.sleep(1)

arduino_info = None
servo_info = None

arduino_lock = threading.Lock()


def sendInfoToArduino():
    data = None
    while True:
        data = arduino_info
        if data is not None:
            print(data)
            arduinoData.write(data.encode())
            time.sleep(0.8)

arduino_thread = threading.Thread(target=sendInfoToArduino)

#arduino_thread.start()


while True:

    with frame_lock:
        if shared_frame is not None:
            frame_for_model = shared_frame.copy()

    image_path = "Object_detection\\image.jpg"
    if frame_for_model is not None:
        cv2.imwrite(image_path,frame_for_model)

        #result = run_detector(detector,image_path)
        #print(result)
        img = cv2.imread(image_path)  # Using OpenCV to load the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        # Perform inference
        results = model(img)
        info = str(results)
        if "person" in info:
            """
                # if person detected then detect who person are
                pred_class = face_model.pred_and_plot(image_path)
                    if pred_class == "Oruç":
                        arduino_info="po"
                    elif pred_class == "Erdem":
                        arduino_info="pe"
                    print(f"{ pred_class} PERSON DETECTED")
            """
            arduino_info = "pe"
            print("PERSON DETECTED")
        else:
            arduino_info = "Not"
            print("PERSON NOT DETECTED")

        
        arduinoData.write(arduino_info.encode())
        time.sleep(1)
        
        
        # Results
        results.print()  # Print results to console


