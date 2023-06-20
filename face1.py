import cv2
import os
import face_recognition
import numpy as np
import sys
import argparse

def find_faces(image_path, scale_factor, min_neighbors, main_img, show_image):
    try:
        base_img = cv2.imread(image_path.strip())
        faces = face_cascade.detectMultiScale(base_img,scale_factor,min_neighbors)
        i=0
        faces_img=[]
        
        if main_img and len(faces) != 1:
            print("can not find any face in the picture or more than one face have been found!")
            return []
                  
        for (x,y,w,h) in faces:
            face = base_img[y:y+h, x:x+w]
            faces_img.insert(i, face)
            i+=1
            if main_img:
                window_name="To find..."
            else:
                window_name="image"+str(i)
                        
        if show_image:    
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, base_img.shape[0:2][1], base_img.shape[0:2][0])
            cv2.imshow(window_name,base_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return [base_img,faces,faces_img]

    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        return []

def idnetify_faces(face_number, person_image_faces, test_image_faces,show_detected):
    scale = 0.3
    results = []
    tm = cv2.TickMeter()

    face_person_image_np = np.asarray(person_image_faces[0],dtype=np.uint8)
    cv2.imwrite(path_save_faces,face_person_image_np)
    person_image_faces_reread = cv2.imread(path_save_faces)

    if scale != 1:
        person_image_faces_reread = cv2.resize(person_image_faces_reread, (int(person_image_faces_reread.shape[1]*scale), int(person_image_faces_reread.shape[0]*scale)))
    
    face_person_image_reread_encoding = face_recognition.face_encodings(person_image_faces_reread)[0]
    os.remove(path_save_faces)

    test_image_faces_np_reread_encoding = []
    for index in range(face_number):
        test_image_faces_np = np.asarray(test_image_faces[index],dtype=np.uint8)
        cv2.imwrite(path_save_faces,test_image_faces_np)
        test_image_faces_np_reread = cv2.imread(path_save_faces)
        if scale != 1:
            test_image_faces_np_reread = cv2.resize(test_image_faces_np_reread, (int(test_image_faces_np_reread.shape[1]*scale), int(test_image_faces_np_reread.shape[0]*scale)))
        test_image_faces_np_reread_encoding.append(face_recognition.face_encodings(test_image_faces_np_reread)[0])
        os.remove(path_save_faces)

    if test_image_faces_np_reread_encoding[0].__len__() > 0:
        tm.start()
        results = face_recognition.compare_faces(test_image_faces_np_reread_encoding, face_person_image_reread_encoding)
        tm.stop()
        index = 0
        for result in results:
            if result:
                window_name = "Image" + str(index+1) + " detected"
            else:
                window_name = "Image" + str(index+1) + " not detected"
            if show_detected:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, test_image_faces[index].shape[0:2][1], test_image_faces[index].shape[0:2][0])
                cv2.imshow(window_name,test_image_faces[index])
            
            index+=1
    
    if show_detected:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("The main proccess took " + str(tm.getTimeMilli()) + " milliseconds")
    return results

def idnetify_faces_movie(captured,person_image_faces):
    base_scale=0.5
    test_scale=0.5
    cv2.namedWindow('the movie', cv2.WINDOW_NORMAL)

    face_person_image_np = np.asarray(person_image_faces[0],dtype=np.uint8)
    cv2.imwrite(path_save_faces,face_person_image_np)
    person_image_faces_reread = cv2.imread(path_save_faces)
    os.remove(path_save_faces)
    
    width = int(person_image_faces_reread.shape[1] * base_scale)
    height = int(person_image_faces_reread.shape[0] * base_scale)
    dim = (width, height)
    person_image_resized = cv2.resize(person_image_faces_reread, dim)
    
    person_image_encoding = face_recognition.face_encodings(person_image_resized)[0]
    _,test_image_org = captured.read()
    while test_image_org is not None:
        if test_scale != 1:
            test_image= cv2.resize(test_image_org, (int(test_image_org.shape[1]*test_scale), int(test_image_org.shape[0]*test_scale)))
        else:
            test_image = test_image_org.copy()

        face_locations = face_recognition.face_locations(test_image)
                
        if face_locations.__len__() > 0:
            faces_encodings = face_recognition.face_encodings(test_image)
            matches = face_recognition.compare_faces(faces_encodings,person_image_encoding)
            i=0
            for match in matches:
                if match:
                    x,y,w,h = face_locations[i]
                    cv2.rectangle(test_image_org,(int(h*(1/test_scale)),int(x*(1/test_scale))),(int(y*(1/test_scale)),int(w*(1/test_scale))),(255,0,0),2)
                    i+=1
        
        cv2.imshow("the movie",test_image_org)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
         
        _,test_image_org = captured.read()
    
def main():
    tm = cv2.TickMeter()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--person_image',dest='person_image', type=str, help='Provide an image to find it in any other images or movies.')
    parser.add_argument('-t','--test_image',dest='test_image', type=str, help='Provide an image to find the person images in it.')
    parser.add_argument('-m','--test_movie',dest='test_movie', type=str, help='Provide an movie to find the person images in it.')
    args = parser.parse_args()
    scale_factor=1.1
    min_neighbors=10
    if not args.person_image:
        print("--person_image is a mandatory argument. you may type -h for more help.")
        # sys.exit(0)
        return False
    else:
        _,_,person_image_faces = find_faces(args.person_image.strip(),scale_factor,min_neighbors,True,True)
        if person_image_faces is None:
            print("The main argument is empty.")
            return False
    
    if args.test_image:
        _,_, test_image_faces = find_faces(args.test_image.strip(),scale_factor,min_neighbors,False,True)
        if test_image_faces is None:
            print("No Face detected in the test image.")
            return False
        
        tm.start()
        results = idnetify_faces(len(test_image_faces), person_image_faces, test_image_faces, True)
        tm.stop()

        index = 0
        i = [ index+1 for found in results if found]
        print("There are {} faces who are identified by the algorithm. face recognition took {} milliseconds".format(i[0],tm.getTimeMilli()))


    if args.test_movie:
        if args.test_movie.strip() == "camera":
            # To capture from camera
            captured = cv2.VideoCapture(0)
        else:
            # To capture from file
            # captured = cv2.VideoCapture("D:\\work\\github\\face-detect\\VID_20150718_134620.mp4")
            captured = cv2.VideoCapture(args.test_movie.strip())
        
        idnetify_faces_movie(captured,person_image_faces)
        captured.release()



cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)
path_save_faces = 'face.jpg'

if __name__ == "__main__":
    main() # pragma: no cover