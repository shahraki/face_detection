import cv2
import os
import face_recognition
import numpy as np
import sys
import argparse

def idnetify_faces_image(face_number,test_faces,img_detected_base,show_detected):
    results=[]
    base_image = np.asarray(img_detected_base[0],dtype=np.uint8)
    cv2.imwrite(path_save_faces,base_image)
    base_image = cv2.imread(path_save_faces)
    base_img_encoding = face_recognition.face_encodings(base_image)[0]
    
    for index in range(face_number):
        face = np.asarray(test_faces[index],dtype=np.uint8)
        cv2.imwrite(path_save_faces,face)
        nface = cv2.imread(path_save_faces)
        
        if face_recognition.face_encodings(nface).__len__() > 0:
            img_to_identify = face_recognition.face_encodings(nface)[0]
            result = face_recognition.compare_faces([base_img_encoding], img_to_identify)
            if result[0]:
                window_name = 'detected'
                results.append(True)
            else:
                window_name = 'not detected'
                results.append(False)
            
            if show_detected:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, nface.shape[0:2][1], nface.shape[0:2][0])
                cv2.imshow(window_name,nface)
    
        os.remove(path_save_faces)

    if show_detected:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results
    

def idnetify_faces_movie(captured,base_img):
    
    aspect=0.5
    _,bigimg = captured.read()
    img = cv2.resize(bigimg, (0, 0), fx=aspect, fy=aspect)
    
    cv2.namedWindow('the movie', cv2.WINDOW_NORMAL)
    base_img_encoding = face_recognition.face_encodings(base_img)[0]
    
    while True:
        face_locations = face_recognition.face_locations(img)
        faces_encodings = face_recognition.face_encodings(img)
        
        if faces_encodings.__len__() > 0:
            matches = face_recognition.compare_faces(faces_encodings,base_img_encoding)
            i=0
            for match in matches:
                if match:
                    x,y,w,h = face_locations[i]
                    cv2.rectangle(bigimg,(int(h*(1/aspect)),int(x*(1/aspect))),(int(y*(1/aspect)),int(w*(1/aspect))),(255,0,0),2)
                    i+=1
    
        cv2.imshow("the movie",bigimg)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
        
        _,bigimg = captured.read()
        if bigimg is None:
            return
        else:
            img = cv2.resize(bigimg, (0, 0), fx=aspect, fy=aspect)

def detect_face(base_img):
    faces = face_cascade.detectMultiScale(base_img,1.1,10)
    i=0
    faces_img=[[0],[0],[0],[0]]
    for (x,y,w,h) in faces:
        face = base_img[y:y+h, x:x+w]
        faces_img.insert(i, face)
        i+=1
    
    return [i,faces_img]

parser = argparse.ArgumentParser()
parser.add_argument('-p','--person_image',dest='person_image', type=str, help='Provide an image to find it in any other images or movies.')
parser.add_argument('-t','--test_image',dest='test_image', type=str, help='Provide an image to find the person images in it.')
parser.add_argument('-m','--test_movie',dest='test_movie', type=str, help='Provide an movie to find the person images in it.')
args = parser.parse_args()

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)
path_save_faces = 'face.jpg'

if not args.person_image:
    print("--person_image is a mandatory argument. you may type -h for more help.")
    sys.exit(0)
else:
    # base_img = cv2.imread("D:\\work\\github\\face-detect\\ID-Card 1.jpg")
    base_img = cv2.imread(args.person_image.strip())
    if base_img.any():
        window_name="To find..."
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, base_img.shape[0:2][1], base_img.shape[0:2][0])
        cv2.imshow(window_name,base_img)
        cv2.waitKey(0)
        _,img_detected_base = detect_face(base_img)
    else:
        print("The main argument is empty.")
        sys.exit(0)

if args.test_image:
    # test_img = cv2.imread("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg")
    test_img = cv2.imread(args.test_image.strip())
    i,imgs_detected_test = detect_face(test_img)
    idnetify_faces_image(i, imgs_detected_test, img_detected_base,True)

if args.test_movie:
    if args.test_movie.strip() == "camera":
        # To capture from camera
        captured = cv2.VideoCapture(0)
    else:
        # To capture from file
        # captured = cv2.VideoCapture("D:\\work\\github\\face-detect\\VID_20150718_134620.mp4")
        captured = cv2.VideoCapture(args.test_movie.strip())
    
    idnetify_faces_movie(captured,base_img)
    captured.release()   


cv2.destroyAllWindows()
