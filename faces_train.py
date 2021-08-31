#import os module for reading training data directories and paths
import os

#import OpenCV module
import cv2 as cv

#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

people = []
for i in os.listdir('Faces/train'):
    people.append(i)

print(people)

DIR = 'Faces/train' #path to the training images
haar_cascade = cv.CascadeClassifier('haar_face.xml') #Instantiate haar cascade

features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY) #Converting image to grayscale

            face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4) #Returns coordinates of the rectangle of the faces detected

            for (x,y,w,h) in face_rect:
                face_roi = gray[y:y+h,x:x+w] #cropping the face
                features.append(face_roi)
                labels.append(label)

# Calling the train functin
create_train()
print(f'Length of features list {len(features)}')
print(f'Length of labels list {len(labels)}')
print("Training done--------------------------")

# Converting into numpy arrays
features = np.array(features,dtype='object')
labels=np.array(labels)

# Instantiationg the face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create() #Local Binary Patterns Histograms

# Train the face recognizer based on features list and labels list
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)