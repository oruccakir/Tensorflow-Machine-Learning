import serial
import time
from FaceModel import FaceModel
import matplotlib.pyplot as plt
import cv2

path = "C:\\Users\\orucc\\Desktop\\oruc_and_erdem"

face_model= FaceModel(path)


arduinoPort = "com7"
arduinoData = serial.Serial(arduinoPort,"9600")
time.sleep(1)

while True:
    
    cam_command = arduinoData.readline()
    cam_command = str(cam_command,'utf-8')
    cam_command = cam_command.strip('\r\n')

    print(cam_command=="0")
    print(cam_command)

    if cam_command == "0":

        image = face_model.capture_the_image()
        
        pred_class = face_model.pred_and_plot(filename="Image Capture and recognition\\Captured_images\\image.jpg")

        text = f"Person is detected ,Estimate  ,{str(pred_class)}"

        print("Estimate : ",str(pred_class))

        face_model.say_the_result(text=text)

        if(pred_class == "Oruç"):
            command = "yellow"
        else:
            command = "red"
        
        arduinoData.write(command.encode())
        
        cv2.imshow("Me",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

