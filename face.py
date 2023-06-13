import cv2
import os
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
                image_to_show = base_img
            else:
                window_name="image"+str(i)
                image_to_show = face
            
            if show_image:    
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, image_to_show.shape[0:2][1], image_to_show.shape[0:2][0])
                cv2.imshow(window_name,image_to_show)
        
        if show_image:   
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return [base_img,faces,faces_img]

    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        return []

def idnetify_faces_image(face_number, test_image, test_faces, base_img, img_detected_base, show_detected,test_detected_img=""):
    results = []
    # recognizer = cv2.FaceRecognizerSF.create("D:\\work\\github\\face-detect\\face_recognition_sface_2021dec_int8.onnx","")
    
    cosine_similarity_threshold = 0.363
    # cosine_similarity_threshold = 0.7
    cosine_score = []
    for index in range(face_number):
        face2_align = []
        face2_feature = []
        window_name="not detected"
        results.append(False)
        
        recognizer = cv2.FaceRecognizerSF.create("D:\\work\\github\\face-detect\\face_recognition_sface_2021dec_int8.onnx","")
        
        face1_align = recognizer.alignCrop(base_img, img_detected_base[0])
        face2_align = recognizer.alignCrop(test_image, test_faces[index])
        
        # Extract features
        face2_feature = recognizer.feature(face2_align)
        face1_feature = recognizer.feature(face1_align)

        cosine_score.append(recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_COSINE))

        if cosine_score[index] >= cosine_similarity_threshold:
            window_name="detected"
            results[index] = True
        if show_detected:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, test_detected_img[index].shape[0:2][1], test_detected_img[index].shape[0:2][0])
            cv2.imshow(window_name,test_detected_img[index])
            cv2.waitKey(0)
    
    
    cv2.destroyAllWindows()

    return results

def main():
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
        base_img,img_detected_base,_ = find_faces(args.person_image.strip(),scale_factor,min_neighbors,True,True)
        if img_detected_base is None:
            print("The main argument is empty.")
            return False
    
    if args.test_image:
        test_image,test_detected, test_detected_img = find_faces(args.test_image.strip(),scale_factor,min_neighbors,False,True)
        if test_detected is None:
            print("No Face detected in the test image.")
            return False
        
        results = idnetify_faces_image(len(test_detected), test_image,test_detected, base_img, img_detected_base,True,test_detected_img)

        i = [ index for found in results if found]
        print("There are {} faces who are identified by the algorithm".format(i))



cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)
path_save_faces = 'face.jpg'

if __name__ == "__main__":
    main() # pragma: no cover