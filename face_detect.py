import cv2
import os
import face_recognition
import numpy as np
import sys
import argparse

def check_path_image(img_path):
    if not img_path or cv2.imread(img_path.strip()) is None:
        return False
    return True

def find_faces(image_path,scale_factor,min_neighbors):
    try:
        base_img = cv2.imread(image_path.strip())
        window_name="To find..."
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, base_img.shape[0:2][1], base_img.shape[0:2][0])
        cv2.imshow(window_name,base_img)
        cv2.waitKey(0)
        return face_cascade.detectMultiScale(base_img,scale_factor,min_neighbors)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

def draw_rectangle(bigimg,faces2,aspect,cosine_score,cosine_similarity_threshold):
    if cosine_score >= cosine_similarity_threshold:
        for (x,y,w,h) in faces2:
            cv2.rectangle(bigimg,(int(x*(1/aspect)),int((y)*(1/aspect))),(int((x+w)*(1/aspect)),int((y+h)*(1/aspect))),(255,0,0),2)

#This function needs the paths of a movie and the image rather than itselve.
def idnetify_faces_movie_new(test_movie,person_image):
    results = True
    aspect=1
    scale_factor=1.1
    min_neighbors=10

    # faces1 = find_faces(person_image,scaleFactor,minNeighbors)

    try:
        base_img = cv2.imread(person_image.strip())
        img1 = cv2.resize(base_img, (0, 0), fx=aspect, fy=aspect)
        window_name="To find..."
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, base_img.shape[0:2][1], base_img.shape[0:2][0])
        cv2.imshow(window_name,base_img)
        cv2.waitKey(0)
        faces1 = face_cascade.detectMultiScale(img1,scale_factor,min_neighbors)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        results = False
    
    cv2.namedWindow('the movie', cv2.WINDOW_NORMAL)
    # To capture from file
    captured = cv2.VideoCapture(test_movie.strip())
    while True:
        _,bigimg = captured.read()
        if bigimg is not None:
            img2 = cv2.resize(bigimg, (0, 0), fx=aspect, fy=aspect)
            
            faces2 = face_cascade.detectMultiScale(img2,scale_factor,min_neighbors)
            recognizer = cv2.FaceRecognizerSF.create("D:\\work\\github\\face-detect\\face_recognition_sface_2021dec_int8.onnx","")
            
            if len(faces2) > 0 :
                face1_align = recognizer.alignCrop(img1, faces1[0])
                face2_align = recognizer.alignCrop(img2, faces2[0])

                # Extract features
                face1_feature = recognizer.feature(face1_align)
                face2_feature = recognizer.feature(face2_align)

                cosine_similarity_threshold = 0.363
                cosine_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_COSINE)
                
                # [cv2.rectangle(bigimg,(int(x*(1/aspect)),int((y)*(1/aspect))),(int((x+w)*(1/aspect)),int((y+h)*(1/aspect))),(255,0,0),2) for (x,y,w,h) in faces2 if cosine_score >= cosine_similarity_threshold]
                draw_rectangle(bigimg,faces2,aspect,cosine_score,cosine_similarity_threshold)
                                            
            cv2.imshow("the movie",bigimg)
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
        else:
            break

    cv2.destroyAllWindows()
    return results

#This function needs the paths of images rather than the images itselve.
def idnetify_faces_image_new(test_faces,img_detected_base,show_detected):
    results=[]

    if not check_path_image(img_detected_base) or not check_path_image(test_faces):
        return [False]
    
    test_img = cv2.imread(test_faces.strip())
    face_number,imgs_detected_test = detect_face(test_img)

    base_img = cv2.imread(img_detected_base.strip())
    _,img_detected_base = detect_face(base_img)
    face = np.asarray(img_detected_base[0],dtype=np.uint8)
    cv2.imwrite(path_save_faces,face)
    mface = cv2.imread(path_save_faces)
    base_img_encoding = face_recognition.face_encodings(mface)[0]
    os.remove(path_save_faces)

    for index in range(face_number):
        face = np.asarray(imgs_detected_test[index],dtype=np.uint8)
        cv2.imwrite(path_save_faces,face)
        nface = cv2.imread(path_save_faces)
        
        img_to_identify = face_recognition.face_encodings(nface)[0]
        if img_to_identify.__len__() > 0:
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
    faces_img=[]
    for (x,y,w,h) in faces:
        face = base_img[y:y+h, x:x+w]
        faces_img.insert(i, face)
        i+=1
    
    return [i,faces_img]

# def check_arguments(args_person_image,args_test_image="",args_test_movie=""):
#     exit_status = True
#     if not args_person_image:
#         print("--person_image is a mandatory argument. you may type -h for more help.")
#         exit_status = False
    
#     img = cv2.imread(args_person_image.strip())
#     if img is None:
#         print("The main argument is empty.")
#         exit_status = False
    
#     return exit_status
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--person_image',dest='person_image', type=str, help='Provide an image to find it in any other images or movies.')
    parser.add_argument('-t','--test_image',dest='test_image', type=str, help='Provide an image to find the person images in it.')
    parser.add_argument('-m','--test_movie',dest='test_movie', type=str, help='Provide an movie to find the person images in it.')
    args = parser.parse_args()

    # cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    # haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier(haar_model)
    # path_save_faces = 'face.jpg'
    ##### idnetify_faces_image_new(args.test_image,args.person_image,False)
    # check_arguments(args.person_image)
    idnetify_faces_movie_new(args.test_movie,args.person_image)
    if not args.person_image:
        print("--person_image is a mandatory argument. you may type -h for more help.")
        sys.exit(0)
        
    else:
        # base_img = cv2.imread("D:\\work\\github\\face-detect\\ID-Card 1.jpg")
        base_img = cv2.imread(args.person_image.strip())
        if base_img is not None and base_img.any():
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

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)
path_save_faces = 'face.jpg'

if __name__ == "__main__":
    main() # pragma: no cover
