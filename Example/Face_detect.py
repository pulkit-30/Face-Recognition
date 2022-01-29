import cv2 as cv

# reading an Image
img = cv.imread("./Faces/Alia Bhatt/6.jpeg")
# showing the original image
cv.imshow("Original Image", img)

#  converting to a gray image
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# showing the gray image
cv.imshow("Gray Image", gray)

#  reading haarcascade_frontalface_default.xml
haar = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
# detecting face and coordinates of the circle
face_rect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
# printing the number of faces detected
print("number of faces detected", len(face_rect))

# drawing a rect over a face
for x, y, w, h in face_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

# showing the image with detected Faces
cv.imshow("Detected Face", img)

cv.waitKey(0)
