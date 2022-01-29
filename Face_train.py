'''
#* Face_train.py
#* Read Image Data from
#* './Faces/' dir and
#* generate trained data
#* libraries - cv2, numpy, os
#* req variables - people, feature, labels, haar_cascade

'''

import cv2 as cv
import numpy as np
import os

'''
#* Defining People Array with Names of the Peoples
#* Will store Person Names whose model has been trained
'''
peoples = []

'''
#* create array of names for each dir inside './Faces/'
#* @ people = ['Alia Bhatt', 'Salman Khan', 'shahrukh Khan']

'''
for name in os.listdir('./Faces'):
    peoples.append(name)

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

# * reading haarcascade_frontalface_default.xml
haar_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

'''
#* function start training for face-recognition
#* by the end it will create
#* an array of features (crop face images mat)
#* an array of labels

'''


def Start_Train_Model():

    # * looping through each name inside peoples array
    for person in peoples:

        # * default path
        path = "./Faces/"
        path = os.path.join(path, person) + '/'

        # * index of current name in peoples array
        label = peoples.index(person)

        # * looping through each image inside './Faces/{person}/'
        # * example each image inside './Faces/Alia Bhatt/'
        for img in os.listdir(path):

            # * creating path
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)

            # * converting rgb image to grayscale image
            gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)

            # * detecting face in the Image
            face_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in face_rect:

                # * croping gray image on the basis of detected face coordinates
                face_roi = gray[y:y+h, x:x+w]

                # * push croped image inside feature array
                feature.append(face_roi)

                # * push label inside labels array
                labels.append(label)


# * calling Start_Train_Model function
Start_Train_Model()

# * converting feature array to numpy object(array)
feature = np.array(feature, dtype='object')

# * converting labels array to numpy array
labels = np.array(labels)


# * creating an instance of open cv build-in face-recognition class
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# * training model using feature and labels data
face_recognizer.train(feature, labels)

# * saving trained model in face_trained.yml file
face_recognizer.save('face_trained.yml')


# * saving features as features.npy file
np.save('features.npy', feature)

# * saving labels as labels.npy file
np.save('labels.npy', labels)
