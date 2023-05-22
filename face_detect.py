import cv2
import os

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)

# To capture from camera
# captured = cv2.VideoCapture(0)

# To capture from file
# captured = cv2.VideoCapture("D:\\user\\Downloads\\pexels-30fps.mp4")
captured = cv2.VideoCapture("D:\\west_world.mp4")


while True:
    _,img = captured.read()
    imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find faces
    faces = face_cascade.detectMultiScale(imgg,1.1,4)
    # draw rectungle 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,h+y),(255,0,0),2)
    cv2.imshow('img', img)
    #stop if scape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

captured.release()
