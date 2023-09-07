import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
import pandas as pd
import keras.models
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

my_model = keras.models.load_model('alphabet_classifier.h5')
alphabet=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

draw_color = (160, 32, 240)
px=0
py=0

result, frame = cap.read()
canvas = np.zeros(frame.shape, np.uint8)
print(frame.shape)

while True:
    result, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgbframe)

    if results.multi_hand_landmarks:
        for single_hand in results.multi_hand_landmarks:
            for id, finger in enumerate(single_hand.landmark):
                # print(id, finger)
                if id==6:
                    jcy = int(finger.y * h)

                if id == 8:
                    cx = int(finger.x * w)
                    cy = int(finger.y * h)
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 0), -1)

                if id == 12:
                    middlehy = int(finger.y * h)

                if id == 9:
                    middlely = int(finger.y * h)

        if middlehy>=middlehy :
            px=0
            py=0
        if middlely<=middlehy:
            if px==0 and py==0:
                px=cx
                py=cy
            cv2.line(frame, (px,py), (cx, cy), draw_color, 130)
            cv2.line(canvas, (px,py), (cx, cy), draw_color, 130)
            px=cx
            py=cy

    canvasgray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvasInverse = cv2.threshold(canvasgray, 50, 255, cv2.THRESH_BINARY_INV)
    canvasInverse = cv2.cvtColor(canvasInverse, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, canvasInverse)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow('Video', frame)
    cv2.imshow('Canvas', canvas)

    displayLetter = ""
    if (cv2.waitKey(1) == ord('p')):
        blank = np.zeros((150,150,3), np.uint8)
        canvas[:150, -150:] = blank
        digitcanvas = cv2.resize(canvas, (28, 28))
        digitcanvas = cv2.cvtColor(digitcanvas, cv2.COLOR_BGR2GRAY)
        digitcanvas = digitcanvas.reshape(1, 28, 28, 1)
        displayLetter = alphabet[np.argmax(my_model.predict(digitcanvas))]
        cv2.putText(canvas, displayLetter, (frame.shape[1]-200,200), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)
        plt.title(displayLetter)
        plt.imshow(digitcanvas.reshape(28, 28), interpolation='nearest', cmap='Greys')
        plt.show()

    if (cv2.waitKey(1) == ord('r')):
        canvas = np.zeros(frame.shape, np.uint8)
    if(cv2.waitKey(1) == ord('q')):
        break

while True:
    print("hello")
    if (cv2.waitKey(1000) == ord('q')):
        break



cv2.destroyAllWindows()