import pyttsx3

def talk(text):
    engine = pyttsx3.init()

    engine.setProperty('voice','HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')

    # Speak the text
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()


# Text you want your computer to say
text_to_speak = "Hello, I am your computer, and I can speak! what are you doing today? I just want to know about you. Tell me everything. Why did you make me talk and what is your name. I love the table and Oruç"

talk(text_to_speak)




