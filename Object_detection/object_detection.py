import cv2
import tensorflow as tf
from object_detection_functions import *
import threading


# load the model that will be used
# get the pretrained model
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

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

while True:

    with frame_lock:
        if shared_frame is not None:
            frame_for_model = shared_frame.copy()

    image_path = "Object_detection\\image.jpg"
    if frame_for_model is not None:
        cv2.imwrite(image_path,frame_for_model)

        result = run_detector(detector,image_path)
        print(result)

