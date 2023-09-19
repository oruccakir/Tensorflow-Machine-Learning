import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Record audio from the microphone for a maximum of 10 seconds
with sr.Microphone() as source:
    print("Please start speaking...")
    recognizer.adjust_for_ambient_noise(source)  # Adjust for noise levels
    audio = recognizer.listen(source, timeout=5)

# Attempt to recognize the speech
try:
    text = recognizer.recognize_google(audio)
    print("You said: " + text)
except sr.UnknownValueError:
    print("Sorry, I couldn't understand your speech.")
except sr.RequestError as e:
    print(f"Could not request results from Google Web Speech API; {e}")

# You can now use the 'text' variable to work with the recognized text.
