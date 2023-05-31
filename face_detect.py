import cv2
import os
import face_recognition
import numpy as np

def idnetify_faces_image(face_number,test_faces,base_image):
    result=[False]
    base_img_encoding = face_recognition.face_encodings(base_image)[0]
    for index in range(face_number):
        face = np.asarray(test_faces[index],dtype=np.uint8)
        cv2.imwrite(path_save_faces,face)
        nface = cv2.imread(path_save_faces)
        
        img_to_identify = face_recognition.face_encodings(nface)[0]
        result = face_recognition.compare_faces([base_img_encoding], img_to_identify)
        
        if result[0]:
            cv2.imshow("detected",nface)
        else:
            cv2.imshow("not detected",nface)
        
        os.remove(path_save_faces)

    cv2.waitKey(0)
    return result

def idnetify_faces_movie(img,faces_test):
    while True:
        imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find faces
        faces = face_cascade.detectMultiScale(imgg,1.1,20)
        i=0
        faces_test=[[0],[0],[0],[0]]
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            faces_test.insert(i, face)
            # cv2.rectangle(test_img,(x,y),(x+w,h+y),(255,0,0),2)
            i+=1

        idnetify_faces_image(i, faces_test, img) 
        #stop if scape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

def detect_face(base_img):
    # base_img_encoding = face_recognition.face_encodings(base_img)[0]
    faces = face_cascade.detectMultiScale(base_img,1.1,10)
    # face_img_base=base_img.copy()
    # for (x,y,w,h) in faces:
    #     face_img_base = face_img_base[y:y+h, x:x+w]
    #     cv2.imshow('Searching for...', face_img_base)
        # cv2.rectangle(base_img,(x,y),(x+w,h+y),(255,0,0),2)
    i=0
    faces_img=[[0],[0],[0],[0]]
    for (x,y,w,h) in faces:
        face = base_img[y:y+h, x:x+w]
        faces_img.insert(i, face)
        # cv2.rectangle(test_img,(x,y),(x+w,h+y),(255,0,0),2)
        i+=1
    
    return [i,faces_img]

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)
path_save_faces = 'face.jpg'
# To capture from camera
# captured = cv2.VideoCapture(0)

# To capture from file
captured = cv2.VideoCapture("D:\\work\\github\\face-detect\\VID_20150718_134620.mp4")
# captured = cv2.VideoCapture("D:\\west_world.mp4")

base_img = cv2.imread("D:\\work\\github\\face-detect\\ID-Card 1.jpg")
test_img = cv2.imread("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg")

_,img_detected_base = detect_face(base_img)
i,imgs_detected_test = detect_face(test_img)

# test_img_encoding = face_recognition.face_encodings(test_img)[0]
# faces = face_cascade.detectMultiScale(test_img,1.1,20)
# i=0
# faces_test=[[0],[0],[0],[0]]
# for (x,y,w,h) in faces:
#     face = test_img[y:y+h, x:x+w]
#     faces_test.insert(i, face)
#     # cv2.rectangle(test_img,(x,y),(x+w,h+y),(255,0,0),2)
#     i+=1
face = np.asarray(img_detected_base[0],dtype=np.uint8)
idnetify_faces_image(i, imgs_detected_test, face)    
_,img = captured.read()
idnetify_faces_movie(img,faces_test)    

captured.release()
# cv2.destroyAllWindows()
