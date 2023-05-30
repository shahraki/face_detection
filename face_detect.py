import cv2
import os
import face_recognition


a = [ [2, 4, 6, 8 ], [ 1, 3, 5, 7 ], [ 8, 6, 4, 2 ], [ 7, 5, 3, 1 ] ] 
          
for i in range(len(a)) : 
    for j in range(len(a[i])) : 
        print(a[i][j], end=" ")
    print()    

b = [[1,2,3,4,5],[6,7,8,9,10]]
print(b[2:5])

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)

# To capture from camera
# captured = cv2.VideoCapture(0)

# To capture from file
# captured = cv2.VideoCapture("D:\\user\\Downloads\\pexels-30fps.mp4")
captured = cv2.VideoCapture("D:\\west_world.mp4")

base_img = cv2.imread("D:\\work\\github\\face-detect\\ID-Card 1.jpg")
cv2.imshow('base_img', base_img)
base_img_encoding = face_recognition.face_encodings(base_img)[0]
faces = face_cascade.detectMultiScale(base_img,1.1,10)
face_img=base_img.copy()
for (x,y,w,h) in faces:
    face_img = face_img[x:x+w,y:y+h,:]
    cv2.rectangle(base_img,(x,y),(x+w,h+y),(255,0,0),2)
cv2.imshow('base_img_faces', base_img)
cv2.imshow('face_img', face_img)


# test_img = cv2.imread("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg")
# test_img_encoding = face_recognition.face_encodings(test_img)[0]
# faces = face_cascade.detectMultiScale(test_img,1.1,4)
# for (x,y,w,h) in faces:
#     cv2.rectangle(test_img,(x,y),(x+w,h+y),(255,0,0),2)
# cv2.imshow('test_img_faces', test_img)

cv2.waitKey(0)
result = face_recognition.compare_faces([base_img_encoding], test_img_encoding)
# cv2.imshow('test_img', test_img)

print("Result: ", result)

# # while True:
#     _,img = captured.read()
#     imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # find faces
#     faces = face_cascade.detectMultiScale(imgg,1.1,4)
#     # draw rectungle 
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,h+y),(255,0,0),2)
#     cv2.imshow('img', img)
#     #stop if scape key is pressed
#     k = cv2.waitKey(30) & 0xff
#     if k==27:
#         break

captured.release()
cv2.destroyAllWindows()
