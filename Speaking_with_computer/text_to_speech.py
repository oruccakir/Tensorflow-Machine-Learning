import pyttsx3

def talk(text):
    engine = pyttsx3.init()

    engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')

    # Speak the text
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()


# Text you want your computer to say

text = "Explore the most advanced text to speech and voice cloning software ever. Create lifelike voiceovers for your content or use our AI voice generator as an easy-to-use text reader.Let your content go beyond text with our advanced Text to Speech tool. Generate high-quality spoken audio in any voice, style, and language. Our text rea"

text_to_speak = text

talk(text_to_speak)




