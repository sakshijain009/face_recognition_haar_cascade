import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml') #Instantiate haar cascade

people = []
for i in os.listdir('Faces/train'):
    people.append(i)

# Instantiationg the face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create() #Local Binary Patterns Histograms

# Reading the yaml file
face_recognizer.read('face_trained.yml')

# Reading a random image
img = cv.imread('Faces/val/madonna/3.jpg')

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detecting the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)

