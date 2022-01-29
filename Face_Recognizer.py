'''
#* Face_Recognizer.py
#* Recognize face using Face_Trained data created using 'Face_train.py'

'''
import cv2 as cv
import numpy as np
import os

haar_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

'''
#* Defining People Array with Names of the Peoples
#* Will store Person Names whose model has been trained
'''
peoples = []

'''
#* create array of names for each dir inside './Faces/'
#* @ people = ['Alia Bhatt', 'Salman Khan', 'shahrukh Khan']

'''
# getting names of a person
for name in os.listdir('./Faces'):
    peoples.append(name)


# * read features.npy file and store in features as numpy array
features = np.load('./features.npy', allow_pickle=True)

# * read labels.npy file and store in labels as numpy array
labels = np.load('./labels.npy')

# * instance of cv build-in face-recognizer class
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# *read face_trained.yml file
face_recognizer.read('./face_trained.yml')

# * sample image to test model
img = cv.imread('./Faces/Alia Bhatt/11.jpeg')

# * convert rgb image to grayscale image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# * detect face
face_rect = haar_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=7)


for (x, y, w, h) in face_rect:

    # * crop recognized face
    face_roi = gray[y:y+h, x:x+w]

    # * predicting the face on the basis of trained model
    # * label - map with peoples
    # * confidence - value similar to accuracy
    label, confidence = face_recognizer.predict(face_roi)
    print("Confidence Value = ", confidence, "Label = ", label)

    # * draw rectangle over the detected face
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    # * string to be put on the Image
    # * person whose image has been detected
    if confidence > 100 or confidence < 30:
        # * write text on the Image
        print('Low Predection !!')
    else:
        # * write text on the Image
        cv.putText(img, str(peoples[label]), (20, 30),
                   cv.FONT_ITALIC, 1.0, (0, 255, 0), thickness=1)

# * Show Image
cv.imshow('Image', img)
cv.waitKey(0)
