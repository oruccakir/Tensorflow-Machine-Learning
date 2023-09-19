import pyttsx3

# Text you want your computer to say
text_to_speak = "Hello, I am your computer, and I can speak! what are you doing today? I just want to know about you. Tell me everything. Why did you make me talk and what is your name"

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set properties (optional)
# You can customize the voice and speech rate, among other things.
# For example, you can set a specific voice using:
#engine.setProperty('voice', 'com.apple.speech.synthesis.voice.Alex')
# Note: Voice names may vary depending on your operating system.

engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')

# Get the available voices
voices = engine.getProperty('voices')

for voice in voices:
    print(voice)
    if "male" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break  # Stop when a male voice is found

# Speak the text
engine.say(text_to_speak)

# Wait for the speech to finish
engine.runAndWait()
