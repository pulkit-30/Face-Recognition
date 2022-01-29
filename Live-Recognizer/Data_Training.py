import numpy as np
import cv2 as cv
import os

persons = np.load('./peoples.npy')

totalFrames = []
totalLabels = []
for name in persons:
    feature = np.load('./'+name+'_feature.npy', allow_pickle=True)
    labels = np.load('./'+name+'_labels.npy')
    totalFrames.append(feature)
    totalLabels.append(labels)

frames = []
labels = []
for list in totalFrames:
    for data in list:
        frames.append(data)

for list in totalLabels:
    for data in list:
        labels.append(data)

frames = np.array(frames, dtype='object')
labels = np.array(labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(frames, labels)
face_recognizer.save('face_trained.yml')
