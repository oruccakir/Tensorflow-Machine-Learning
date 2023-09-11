import pyaudio
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

path = "C:\\Users\\orucc\\Desktop\\Coding_Projects\\Tensorflow Machine Learning\\Tensorflow-Machine-Learning-1\\Audio\\recognize_keywords"
model = tf.keras.models.load_model(path)
print(model.summary())

label_names = np.array(['down', 'go', 'left', 'no', 'right', 'stop', 'up' ,'yes'])
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
RECORD_SECONDS = 2

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



        # Ses akışını kapatyes
        stream.stop_stream()
        stream.close()
        audio.terminate()

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