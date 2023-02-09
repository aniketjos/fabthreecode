import cv2
fcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #we required trained data for face detection in opencv
img = cv2.imread("face_detection_1.jpg")
from matplotlib import pyplot as plt
plt.imshow(img)
color_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #for showing proper colour RGB image
plt.imshow(color_img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #we convert the colour image into gray image
plt.imshow(gray)
faces = fcascade.detectMultiScale(gray, 1.1, 9) # scalefactor =1.1 for fixed zoom value because all image have different value and 9 is minNeighbour value Parameter specifying how many neighbors each candidate rectangle should have to retain it. 
# Iterating through rectangles of detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  #here it draw the rectangle around the detected faces
cv2.imshow('Detected faces', img)
cv2.waitKey(0)
