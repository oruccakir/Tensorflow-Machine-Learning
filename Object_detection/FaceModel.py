import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
import pyttsx3

class_names = ["Erdem","Oruç"]

class FaceModel:

    def __init__(self,model_path,class_names=class_names):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names


    def print_model_info(self):
        self.model.summary()


    def load_and_prep_image(self,filename, img_shape=224):
        """
        Reads an image from filename, turns it into a tensor
        and reshapes it to (img_shape, img_shape, colour_channel).
        """
        # Read in target file (an image)
        img = tf.io.read_file(filename)

        # Decode the read file into a tensor & ensure 3 colour channels
        # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
        img = tf.image.decode_image(img, channels=3)

        # Resize the image (to the same size our model was trained on)
        img = tf.image.resize(img, size = [img_shape, img_shape])

        # Rescale the image (get all values between 0 and 1)
        return img
    

    def pred_and_plot(self, filename, class_names = class_names):
        """
        Imports an image located at filename, makes a prediction on it with
        a trained model and plots the image with the predicted class as the title.
        """
        # Import the target image and preprocess it
        img = self.load_and_prep_image(filename=filename)

        # Make a prediction
        pred = self.model.predict(tf.expand_dims(img, axis=0))

        # Get the predicted class
        pred_class = class_names[int(tf.round(pred)[0][0])]

        # Plot the image and predicted class
        #plt.imshow(img/255)
        #plt.title(f"Prediction: {pred_class}")
        #plt.axis(False);
        print(pred_class)

        return pred_class


    
    def capture_the_image(self):
        # Initialize the camera
        cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
        else:
            # Capture a single frame
            ret, frame = cap.read()

            if ret:
                # Save the captured frame to a file
                #cv2.imwrite("captured_image.jpg", frame)
                #print("Image captured and saved as 'captured_image.jpg'")
                cv2.imwrite("Image Capture and recognition\\Captured_images\\image.jpg",frame)
                return frame
            else:
                return None
            
    
    def say_the_result(self,text):
        engine = pyttsx3.init()

        engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')


        # Speak the text
        engine.say(text)

        # Wait for the speech to finish
        engine.runAndWait()
