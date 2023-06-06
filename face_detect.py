import cv2
import os
import face_recognition
import numpy as np

def idnetify_faces_image(face_number,test_faces,img_detected_base,show_detected):
    results=[]
    base_image = np.asarray(img_detected_base[0],dtype=np.uint8)
    cv2.imwrite(path_save_faces,base_image)
    base_image = cv2.imread(path_save_faces)
    base_img_encoding = face_recognition.face_encodings(base_image)[0]
    
    # if show_detected:
    #     window_name = 'detected'
    # else:
    #     window_name = 'not detected'
    
    for index in range(face_number):
        face = np.asarray(test_faces[index],dtype=np.uint8)
        cv2.imwrite(path_save_faces,face)
        nface = cv2.imread(path_save_faces)
        
        # img_to_identify = encode_face(nface)
        if face_recognition.face_encodings(nface).__len__() > 0:
            img_to_identify = face_recognition.face_encodings(nface)[0]
        # if img_to_identify.__len__() > 0 :
            result = face_recognition.compare_faces([base_img_encoding], img_to_identify)
            if result[0]:
                window_name = 'detected'
                results.append(True)
                
            else:
                window_name = 'not detected'
                results.append(False)
            
            if show_detected:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, nface.shape[0:2][0], nface.shape[0:2][1])
                cv2.imshow(window_name,nface)
    
        os.remove(path_save_faces)

    if show_detected:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results
    

def idnetify_faces_movie(captured,base_img):
    
    aspect=0.75
    _,bigimg = captured.read()
    img = cv2.resize(bigimg, (0, 0), fx=aspect, fy=aspect)
    # cv2.imwrite(path_save_faces,img)
    # img = cv2.imread(path_save_faces)
    # os.remove(path_save_faces)

    cv2.namedWindow('the movie', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('the movie', bigimg.shape[0:2][0], bigimg.shape[0:2][1])
    base_img_encoding = face_recognition.face_encodings(base_img)[0]
    
    while True:
        # rgb_img = img[:, :, ::-1]
        # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(img)

        faces_encodings = face_recognition.face_encodings(img)
        # imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find faces
        # faces = face_cascade.detectMultiScale(imgg,1.1,4)
        # nfaces = np.asarray(faces,dtype=np.uint8)

        
        # _,img_detected_base = detect_face(base_img)
        # i,imgs_detected_test = detect_face(imgg)
        # results = idnetify_faces_image(i,imgs_detected_test,img_detected_base,False)
        if faces_encodings.__len__() > 0:
            matches = face_recognition.compare_faces(faces_encodings,base_img_encoding)
            i=0
            for match in matches:
                if match:
                    x,y,w,h = face_locations[i]
                    cv2.rectangle(bigimg,(int(h*(1/aspect)),int(x*(1/aspect))),(int(y*(1/aspect)),int(w*(1/aspect))),(255,0,0),2)
                    i+=1

        # i=0
        # # faces_test=[[0],[0],[0],[0]]
        # for (x,y,w,h) in face_locations:
        #     # face = imgg[y:y+h, x:x+w]
        #     # faces_test.insert(i, face)
        #     # face_encoding = face_recognition.face_encodings(face)[0]

        #     if faces_encodings.__len__() > 0 and face_recognition.compare_faces([base_img_encoding], faces_encodings[i]).__contains__(True):
        #             matches = face_recognition.compare_faces(faces_encodings,base_img_encoding)
        #             # cv2.rectangle(img,(x,y),(x+w,h+y),(255,0,0),2)
        #             # cv2.rectangle(bigimg,(h*(1/aspect),x*(1/aspect)),(y*(1/aspect),w*(1/aspect)),(255,0,0),2)
        #             cv2.rectangle(bigimg,(int(h*(1/aspect)),int(x*(1/aspect))),(int(y*(1/aspect)),int(w*(1/aspect))),(255,0,0),2)
        #     i+=1
        #     # for res in results:
        #     #     if res:
        #     #         cv2.rectangle(img,(x,y),(x+w,h+y),(255,0,0),2)
        #     # i+=1
            
        
        cv2.imshow("the movie",bigimg)
        # idnetify_faces_image(i, faces_test, img) 
        #stop if scape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
        
        _,bigimg = captured.read()
        if bigimg is None:
            return
        else:
            img = cv2.resize(bigimg, (0, 0), fx=aspect, fy=aspect)
            
        # cv2.imwrite(path_save_faces,img)
        # img = cv2.imread(path_save_faces)
        # os.remove(path_save_faces)

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

idnetify_faces_image(i, imgs_detected_test, img_detected_base,True)    

# _,img = captured.read()
idnetify_faces_movie(captured,base_img)    

captured.release()
cv2.destroyAllWindows()
