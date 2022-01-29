import cv2 as cv
import numpy as np
import os

'''
#* Defining People Array with Names of the Peoples
#* Will store Person Names whose model has been trained
'''
peoples = []

# getting names of a person
for name in os.listdir('./Faces'):
    peoples.append(name)

print(peoples)
'''
#* defining feature array
#* will store all the features for an Image

'''
feature = []

'''
#* defining Labels array
#* will store int value that map in for PersonName in `peoples` array

'''
labels = []

haar_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

x = 0


def Start_Train_Model():
    for person in peoples:
        path = "./Faces/"
        path = os.path.join(path, person)+'/'
        label = peoples.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)

            # detecting face in the Image
            face_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in face_rect:
                face_roi = gray[y:y+h, x:x+w]
                feature.append(face_roi)
                labels.append(label)


Start_Train_Model()

feature = np.array(feature, dtype='object')
labels = np.array(labels)


#
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(feature, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', feature)
np.save('labels.npy', labels)
