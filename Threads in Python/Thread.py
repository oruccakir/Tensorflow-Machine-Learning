from time import *
from threading import Thread

def myBox():
    while True:
        print("My box is open")
        sleep(5)
        print("My box is closed")
        sleep(5)

def myLED():
    while True:
        print("My LED is open")
        sleep(1)
        print("My LED is off")
        sleep(1)


boxThread = Thread(target=myBox)
ledThread = Thread(target=myLED)


