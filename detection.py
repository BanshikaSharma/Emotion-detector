import warnings
import numpy as np
import cv2
import deepface
from deepface import DeepFace
import pyttsx3
import speech_recognition as sr
import time
import tkinter as tk
from tkinter import messagebox


cap = cv2.VideoCapture(0)


engine = pyttsx3.init()


window = tk.Tk()
window.withdraw()


emotion_positions = {
    'angry': (10, 30),
    'disgusted': (10, 50),
    'fearful': (10, 70),
    'happy': (10, 90),
    'neutral': (10, 110),
    'sad': (10, 130),
    'surprised': (10, 150)
}

while True:
  
    ret, frame = cap.read()
    frame = np.array(frame)
    print(frame)

   
    try:
        emotions = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)[0]['emotion']
        emotion_key = max(emotions, key=emotions.get)
        emotion_value = emotions[emotion_key]
        cv2.putText(frame, str(emotion_key + ': ' + str(round(emotion_value * 100, 2)) + '%'), emotion_positions[emotion_key], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    except KeyError as e:
        print("Could not find emotion position for: " + str(e))

   
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = sr.listen(source)
            text = sr.recognize_google(audio)
            print("You said: " + str(text))

            print("Speaking...")
            engine.say(str(text))
            engine.runAndWait()
    except Exception as e:
        print("Sorry could not recognize what you said:", str(e))
        text = ""

   
    if text != "":
        messagebox.showinfo("Speech Recognition", str(text))

    
    cv2.imshow('Facial Expression and Speech Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
