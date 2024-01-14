from ImageModel import ImageModel
import matplotlib.pyplot as plt
import cv2

path = "Image Capture and recognition\\Saved_Models\\domainnet_without_mixed_precision"

image_model= ImageModel(path)

while True:

    image_model.say_the_result("Do you want to show image to predict ?")

    command = input("Do you want to show image to predict ?")

    if command == "1":
        image = image_model.capture_the_image()
        
        pred_class = image_model.pred_and_plot_saved(filename="Image Capture and recognition\\Captured_images\\image.jpg")

        text = f"Object is detected Estimate  {str(pred_class)}"

        print("Estimate : ",str(pred_class))

        image_model.say_the_result(text=text)

        cv2.imshow("Me",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
       break