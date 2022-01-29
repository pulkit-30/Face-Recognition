import cv2 as cv
import numpy as np
import os

haar_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

'''
#* Defining People Array with Names of the Peoples
#* Will store Person Names whose model has been trained
'''
peoples = []

# getting names of a person
for name in os.listdir('./Faces'):
    peoples.append(name)


features = np.load('./features.npy', allow_pickle=True)
labels = np.load('./labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read('./face_trained.yml')


img = cv.imread('./mis2.jpeg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detect face
face_rect = haar_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in face_rect:
    face_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    # if confidence > 100:
    #     continue
    cv.imshow("Sample", face_roi)
    print(confidence, label)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.putText(img, str(peoples[label]), (20, 30),
               cv.FONT_ITALIC, 1.0, (0, 255, 0), thickness=1)


cv.imshow('Image', img)
cv.waitKey(0)
