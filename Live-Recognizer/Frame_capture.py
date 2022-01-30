import cv2 as cv
import numpy as np
import os

feature = []
labels = []
haar = cv.CascadeClassifier('../haarcascade_frontalface_default.xml')
# take name as input
persons = np.load('./peoples.npy')
name = input('Enter Your Good Name')
persons = np.append(persons, name)


capture = cv.VideoCapture(0)

time = 0
while time < 100:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    face_rect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    print("number of faces detected --> ", len(face_rect))
    for x, y, w, h in face_rect:
        face_roi = gray[y:y+h, x:x+w]
        feature.append(face_roi)
        labels.append(len(persons)-1)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    time += 1
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

if len(feature) > 50:
    feature = np.array(feature, dtype='object')
    labels = np.array(labels)

    np.save(name+'_feature.npy', feature)
    np.save(name+'_labels.npy', labels)
    np.save('peoples.npy', persons)
    print('----_Data Capture Successfully🥳_----')
else:
    print("Cannot Capture your face data!!!!")
