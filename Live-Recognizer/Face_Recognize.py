
import cv2 as cv
import numpy as np
import os

haar = cv.CascadeClassifier('../haarcascade_frontalface_default.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('./face_trained.yml')

peoples = np.load('./peoples.npy')

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    face_rect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    print("number of faces detected --> ", len(face_rect))
    for x, y, w, h in face_rect:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_roi)
        print("Confidence Value = ", confidence, "Label = ", label)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        if confidence > 100 or confidence < 10:
            print('Low Prediction !!')
        else:
            cv.putText(frame, str(peoples[label]), (20, 30),
                       cv.FONT_ITALIC, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
