from FaceModel import FaceModel
import matplotlib.pyplot as plt
import cv2

path = "C:\\Users\\orucc\\Desktop\\oruc_and_erdem"

face_model= FaceModel(path)


while True:

    face_model.say_the_result("Show me yourself, by looking at, the camera?")

    command = input("Enter command : ")

    if command == "1":
        image = face_model.capture_the_image()
        
        pred_class = face_model.pred_and_plot(filename="Image Capture and recognition\\Captured_images\\image.jpg")

        text = f"Person is detected ,Estimate  ,{str(pred_class)}"

        print("Estimate : ",str(pred_class))

        face_model.say_the_result(text=text)

        cv2.imshow("Me",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
       break