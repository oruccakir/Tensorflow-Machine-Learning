import pyaudio
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

try:
    from vpython import *
except ZeroDivisionError:
    print()

# Küreyi oluştur
ball = sphere(pos=vector(0, 0, 0), radius=0.4, color=color.blue)

# Hareket hızı
move_speed = 1

path = "C:\\Users\\orucc\\Desktop\\Coding_Projects\\Tensorflow Machine Learning\\Tensorflow-Machine-Learning-1\\Audio\\recognize_keyword_with_more_data"
model = tf.keras.models.load_model(path)
print(model.summary())


label_names = np.array(['down','go','left','no','off','on','right','stop','up','yes','_silence_','_unknown_'])

#label_names = ['down', 'go' ,'left', 'no', 'right', 'stop', 'up' ,'yes']


print(label_names)

def get_spectrogram(waveform):
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



CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1

audio = pyaudio.PyAudio()


while(True):

    command = input("Enter the command :")

    if command == "yes":

        # Ses akışını başlat
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
        print("Shape of data : " ,audio_data.shape)
        audio_data = tf.constant(audio_data)
        print(audio_data)

        audio_data_spec = get_spectrogram(audio_data)
        audio_data_spec = audio_data_spec[tf.newaxis,...]

        print('Waveform shape:', audio_data.shape)
        print('Spectrogram shape:', audio_data_spec.shape)

        guess = model.predict(audio_data_spec)
        guess = tf.squeeze(tf.round(guess))
        guess = tf.argmax(guess)
        guess = guess.numpy()
        print(guess)
        print(label_names[guess])
        direction = label_names[guess]



        # Ses akışını kapatyes
        stream.stop_stream()
        stream.close()
        audio.terminate()

        if direction == "left":
            ball.pos.x -= move_speed
        elif direction == "right":
            ball.pos.x += move_speed
        elif direction == "up":
            ball.pos.y += move_speed
        elif direction == "down":
            ball.pos.y -= move_speed
        elif direction == "go":
            ball.pos.z -= move_speed
        elif direction == "yes":
            ball.pos.z += move_speed

        # Ses verisini çal
        audio_play = pyaudio.PyAudio()
        play_stream = audio_play.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, output=True)

        print("Playing recorded audio...")

        for frame in frames:
            play_stream.write(frame)

        print("Finished playing audio.")



        play_stream.stop_stream()
        play_stream.close()

    else:
        break
        
