import pyaudio
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# import necessary libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

import tensorflow as tf
import tensorflow_io as tfio

import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile

import numpy as np
from scipy.io import wavfile

def load_the_model(url="https://tfhub.dev/google/yamnet/1"):
    # Load the model
    model = hub.load(url)
    return model

def play_the_voice(filename):

    wav_file_name = filename
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # Show some basic information about the audio.
    duration = len(wav_data)/sample_rate
    print(f'Sample rate: {sample_rate} Hz')
    print(f'Total duration: {duration:.2f}s')
    print(f'Size of the input: {len(wav_data)}')

    return wav_data
  
def normalize_the_data(wav_data):
   return wav_data / tf.int16.max

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


def execute_the_model(model,waveform):
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    print(f'The main sound is: {infered_class}')

    return infered_class


url = "https://tfhub.dev/google/yamnet/1"

model = load_the_model(url=url)

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)


wav_data = play_the_voice("Audio\\speech_whistling2.wav")

waveform = normalize_the_data(wav_data)

print(waveform)

waveform = np.array(waveform)

waveform = waveform.astype(np.float32)

print(waveform)

execute_the_model(model,waveform)


CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

audio = pyaudio.PyAudio()

print(class_names)

while True:
    command = input("Enter the command :")

    if command == "yes":
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

        execute_the_model(model,audio_data)

    else:
       break