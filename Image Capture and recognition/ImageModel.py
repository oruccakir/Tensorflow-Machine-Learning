import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
import pyttsx3

label_names = ['aircraft_carrier',
 'airplane',
 'alarm_clock',
 'ambulance',
 'angel',
 'animal_migration',
 'ant',
 'anvil',
 'apple',
 'arm',
 'asparagus',
 'axe',
 'backpack',
 'banana',
 'bandage',
 'barn',
 'baseball',
 'baseball_bat',
 'basket',
 'basketball',
 'bat',
 'bathtub',
 'beach',
 'bear',
 'beard',
 'bed',
 'bee',
 'belt',
 'bench',
 'bicycle',
 'binoculars',
 'bird',
 'birthday_cake',
 'blackberry',
 'blueberry',
 'book',
 'boomerang',
 'bottlecap',
 'bowtie',
 'bracelet',
 'brain',
 'bread',
 'bridge',
 'broccoli',
 'broom',
 'bucket',
 'bulldozer',
 'bus',
 'bush',
 'butterfly',
 'cactus',
 'cake',
 'calculator',
 'calendar',
 'camel',
 'camera',
 'camouflage',
 'campfire',
 'candle',
 'cannon',
 'canoe',
 'car',
 'carrot',
 'castle',
 'cat',
 'ceiling_fan',
 'cello',
 'cell_phone',
 'chair',
 'chandelier',
 'church',
 'circle',
 'clarinet',
 'clock',
 'cloud',
 'coffee_cup',
 'compass',
 'computer',
 'cookie',
 'cooler',
 'couch',
 'cow',
 'crab',
 'crayon',
 'crocodile',
 'crown',
 'cruise_ship',
 'cup',
 'diamond',
 'dishwasher',
 'diving_board',
 'dog',
 'dolphin',
 'donut',
 'door',
 'dragon',
 'dresser',
 'drill',
 'drums',
 'duck',
 'dumbbell',
 'ear',
 'elbow',
 'elephant',
 'envelope',
 'eraser',
 'eye',
 'eyeglasses',
 'face',
 'fan',
 'feather',
 'fence',
 'finger',
 'fire_hydrant',
 'fireplace',
 'firetruck',
 'fish',
 'flamingo',
 'flashlight',
 'flip_flops',
 'floor_lamp',
 'flower',
 'flying_saucer',
 'foot',
 'fork',
 'frog',
 'frying_pan',
 'garden',
 'garden_hose',
 'giraffe',
 'goatee',
 'golf_club',
 'grapes',
 'grass',
 'guitar',
 'hamburger',
 'hammer',
 'hand',
 'harp',
 'hat',
 'headphones',
 'hedgehog',
 'helicopter',
 'helmet',
 'hexagon',
 'hockey_puck',
 'hockey_stick',
 'horse',
 'hospital',
 'hot_air_balloon',
 'hot_dog',
 'hot_tub',
 'hourglass',
 'house',
 'house_plant',
 'hurricane',
 'ice_cream',
 'jacket',
 'jail',
 'kangaroo',
 'key',
 'keyboard',
 'knee',
 'knife',
 'ladder',
 'lantern',
 'laptop',
 'leaf',
 'leg',
 'light_bulb',
 'lighter',
 'lighthouse',
 'lightning',
 'line',
 'lion',
 'lipstick',
 'lobster',
 'lollipop',
 'mailbox',
 'map',
 'marker',
 'matches',
 'megaphone',
 'mermaid',
 'microphone',
 'microwave',
 'monkey',
 'moon',
 'mosquito',
 'motorbike',
 'mountain',
 'mouse',
 'moustache',
 'mouth',
 'mug',
 'mushroom',
 'nail',
 'necklace',
 'nose',
 'ocean',
 'octagon',
 'octopus',
 'onion',
 'oven',
 'owl',
 'paintbrush',
 'paint_can',
 'palm_tree',
 'panda',
 'pants',
 'paper_clip',
 'parachute',
 'parrot',
 'passport',
 'peanut',
 'pear',
 'peas',
 'pencil',
 'penguin',
 'piano',
 'pickup_truck',
 'picture_frame',
 'pig',
 'pillow',
 'pineapple',
 'pizza',
 'pliers',
 'police_car',
 'pond',
 'pool',
 'popsicle',
 'postcard',
 'potato',
 'power_outlet',
 'purse',
 'rabbit',
 'raccoon',
 'radio',
 'rain',
 'rainbow',
 'rake',
 'remote_control',
 'rhinoceros',
 'rifle',
 'river',
 'roller_coaster',
 'rollerskates',
 'sailboat',
 'sandwich',
 'saw',
 'saxophone',
 'school_bus',
 'scissors',
 'scorpion',
 'screwdriver',
 'sea_turtle',
 'see_saw',
 'shark',
 'sheep',
 'shoe',
 'shorts',
 'shovel',
 'sink',
 'skateboard',
 'skull',
 'skyscraper',
 'sleeping_bag',
 'smiley_face',
 'snail',
 'snake',
 'snorkel',
 'snowflake',
 'snowman',
 'soccer_ball',
 'sock',
 'speedboat',
 'spider',
 'spoon',
 'spreadsheet',
 'square',
 'squiggle',
 'squirrel',
 'stairs',
 'star',
 'steak',
 'stereo',
 'stethoscope',
 'stitches',
 'stop_sign',
 'stove',
 'strawberry',
 'streetlight',
 'string_bean',
 'submarine',
 'suitcase',
 'sun',
 'swan',
 'sweater',
 'swing_set',
 'sword',
 'syringe',
 'table',
 'teapot',
 'teddy-bear',
 'telephone',
 'television',
 'tennis_racquet',
 'tent',
 'The_Eiffel_Tower',
 'The_Great_Wall_of_China',
 'The_Mona_Lisa',
 'tiger',
 'toaster',
 'toe',
 'toilet',
 'tooth',
 'toothbrush',
 'toothpaste',
 'tornado',
 'tractor',
 'traffic_light',
 'train',
 'tree',
 'triangle',
 'trombone',
 'truck',
 'trumpet',
 't-shirt',
 'umbrella',
 'underwear',
 'van',
 'vase',
 'violin',
 'washing_machine',
 'watermelon',
 'waterslide',
 'whale',
 'wheel',
 'windmill',
 'wine_bottle',
 'wine_glass',
 'wristwatch',
 'yoga',
 'zebra',
 'zigzag']


class ImageModel:
    def __init__(self,model_path,class_names=label_names):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
    
    def print_model_info(self):
        self.model.summary()

    def load_and_prep_image(self,img, img_shape=224,scale=False):
        """
        Reads in an image from filename, turns it into a tensor and reshapes into specified shape
        (img_shape,img_shape,color_channels=3).

        Args:
            filename (str) : path to target image
            img_shape (int) : height/width dimension of the target image size
            scale (bool) : Scale pixel values from 0-255 to 0-1 or not

        Returns
            Image tensor of shape (img_shape,img_shape,3)
        """
        # Decode image into a tensor
        #img = tf.io.decode_image(img, channels=3)

        # Resize the image
        img = tf.image.resize(img,[img_shape,img_shape])

        # Scale? Yes/No
        if scale:
            # rescale the image (get all values betweem 0 & 1)
            return img/255.
        else:
            return img # don't need to rescale images for EfficientNet models in TensorFlow
        

    def pred_and_plot(self,img,class_names = label_names):
        """
        Imports an image located at filename, makes a prediction on it with
        a trained model and plots the image with the predicted class as the title.
        """
        # Import the target image and preprocess it
        img = self.load_and_prep_image(img,scale=False)

        # Make a prediction
        pred = self.model.predict(tf.expand_dims(img, axis=0))

        # Get the predicted class
        if len(pred[0]) > 1: # check for multi-class
            pred_class = class_names[pred.argmax()] # if more than one output, take the max
        else:
            pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

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

    
    def pred_and_plot_saved(self,filename,class_names = label_names):
        """
        Imports an image located at filename, makes a prediction on it with
        a trained model and plots the image with the predicted class as the title.
        """
        # Import the target image and preprocess it
        img = self.load_and_prep_image_saved(filename,scale=False)

        # Make a prediction
        pred = self.model.predict(tf.expand_dims(img, axis=0))

        # Get the predicted class
        if len(pred[0]) > 1: # check for multi-class
            pred_class = class_names[pred.argmax()] # if more than one output, take the max
        else:
            pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

        return pred_class

    # Create a function to load and prepare images
    def load_and_prep_image_saved(self,filename, img_shape=224,scale=False):
        """
        Reads in an image from filename, turns it into a tensor and reshapes into specified shape
        (img_shape,img_shape,color_channels=3).

        Args:
            filename (str) : path to target image
            img_shape (int) : height/width dimension of the target image size
            scale (bool) : Scale pixel values from 0-255 to 0-1 or not

        Returns
            Image tensor of shape (img_shape,img_shape,3)
        """
        # Read in the image
        img = tf.io.read_file(filename=filename)

        # Decode image into a tensor
        img = tf.io.decode_image(img, channels=3)

        # Resize the image
        img = tf.image.resize(img,[img_shape,img_shape])

        # Scale? Yes/No
        if scale:
            # rescale the image (get all values betweem 0 & 1)
            return img/255.
        else:
            return img # don't need to rescale images for EfficientNet models in TensorFlow
