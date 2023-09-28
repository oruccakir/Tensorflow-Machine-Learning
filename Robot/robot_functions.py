# import necessary libraries to define and execute the methods

# for capturing image import cv2 library
import cv2
# for image processing and tensors import tensorflow
import tensorflow as tf
# for speech to text or text to speech import pyttsx3
import pyttsx3
# for recording the auido import pyaudio
import pyaudio
# for data processing import nump
import numpy as np

def preprocess_image(img, img_shape=224,scale=False):
    """
    In models efficientBX base models are used as feature extractor or used as fine-tuning therefore there is no need to scale, 
    turns it into a tensor and reshapes into specified shape
    (img_shape,img_shape,color_channels=3).

    Args:
        img : path target image
        img_shape (int) : height/width dimension of the target image size
        scale (bool) : Scale pixel values from 0-255 to 0-1 or not

    Returns
        Image tensor of shape (img_shape,img_shape,3)
    """

    # Resize the image
    img = tf.image.resize(img,[img_shape,img_shape])

    # Scale? Yes/No
    if scale:
        # rescale the image (get all values betweem 0 & 1)
        return img/255.
    else:
        return img # don't need to rescale images for EfficientNet models in TensorFlow
        





def make_prediction_on_image(model,img,class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.

    Returns predictes class as string

    """
    # Import the target image and preprocess it
    img = model.load_and_prep_image(img,scale=False)

    # Make a prediction
    pred = model.model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1: # check for multi-class
        pred_class = class_names[pred.argmax()] # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

    return pred_class





def capture_the_image():
    """
    Captures an image from camera and returns it
    """
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)
        
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    else:
        # Capture a single frame
        ret, frame = cap.read()

        if ret:
            return frame
        else:
            return None
        



def capture_the_audio_and_make_predictions_on_the_audio(model,class_names):

    """
    Captures audio with below parameters and makes predictions on that audio and returns class names as string
    """

    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 2

    audio = pyaudio.PyAudio()

    # start voice flow
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
    audio_data = np.array(audio_data)

    audio_data = tf.constant(audio_data)
    

    audio_data_spec = get_spectrogram(audio_data)
    audio_data_spec = audio_data_spec[tf.newaxis,...]

    
    guess = model.predict(audio_data_spec)
    guess = tf.squeeze(tf.round(guess))
    guess = tf.argmax(guess)
    guess = guess.numpy()


    # close voice flow
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return class_names[guess]



def get_spectrogram(waveform):
  """
  Get waveforms and convert them into spectograms and return spectograms
  """
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  spectrogram = tf.image.resize(spectrogram, (124,129))
  return spectrogram




def save_the_captured_image(frame,given_path,image_name):
    """
    Takes image as an input and then save it to given path under the given name
    Not : it accepts its path as a full not relative
    """
    given_path = given_path + "\\"+ image_name +".jpg"

    cv2.imwrite(given_path,frame)




def show_the_captured_image(name,image):
    """
    Takes a name and image and show the image on the screen
    """
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def say_the_text(text):
    """
    Uses text to speech method and says the text
    """

    talk_engine = pyttsx3.init()

    talk_engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')


    # Speak the text
    talk_engine.say(text)

    # Wait for the speech to finish
    talk_engine.runAndWait()


