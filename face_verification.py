import cv2
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p','--person_image',dest='person_image', type=str, help='Provide an image to find it in any other images or movies.')
parser.add_argument('-t','--test_image',dest='test_image', type=str, help='Provide an image to find the person images in it.')
parser.add_argument('-m','--test_movie',dest='test_movie', type=str, help='Provide an movie to find the person images in it.')
args = parser.parse_args()

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)
path_save_faces = 'face.jpg'
scaleFactor=1.1
minNeighbors=10


if not args.person_image:
    print("--person_image is a mandatory argument. you may type -h for more help.")
    sys.exit(0)
else:
    base_img = cv2.imread(args.person_image.strip())
    if base_img.any():
        window_name="To find..."
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, base_img.shape[0:2][1], base_img.shape[0:2][0])
        cv2.imshow(window_name,base_img)
        cv2.waitKey(0)
        faces1 = face_cascade.detectMultiScale(base_img,scaleFactor,minNeighbors)
    else:
        print("The main argument is empty.")
        sys.exit(0)

if args.test_movie:
    if args.test_movie.strip() == "camera":
        # To capture from camera
        captured = cv2.VideoCapture(0)
    else:
        cv2.namedWindow('the movie', cv2.WINDOW_NORMAL)
        # To capture from file
        # captured = cv2.VideoCapture("D:\\work\\github\\face-detect\\VID_20150718_134620.mp4")
        captured = cv2.VideoCapture(args.test_movie.strip())
        aspect=0.75
        while True:
            _,bigimg = captured.read()
            if bigimg is not None:
                img2 = cv2.resize(bigimg, (0, 0), fx=aspect, fy=aspect)
                
                faces2 = face_cascade.detectMultiScale(img2,scaleFactor,minNeighbors)
                # detector = cv2.FaceDetectorYN.create(haar_model,"",(320, 320),0.9,0.3,5000)
                recognizer = cv2.FaceRecognizerSF.create("D:\\work\\github\\face-detect\\face_recognition_sface_2021dec_int8.onnx","")
                # recognizer = cv2.FaceRecognizerSF.create("D:\\work\\github\\face-detect\\face_recognition_sface_2021dec.onnx","")
                

                if len(faces2) > 0 :
                    face1_align = recognizer.alignCrop(base_img, faces1[0])
                    face2_align = recognizer.alignCrop(img2, faces2[0])

                    # Extract features
                    face1_feature = recognizer.feature(face1_align)
                    face2_feature = recognizer.feature(face2_align)

                    cosine_similarity_threshold = 0.363
                    l2_similarity_threshold = 1.128

                    cosine_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    l2_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_NORM_L2)

                    msg = 'different identities'
                    if cosine_score >= cosine_similarity_threshold:
                        msg = 'the same identity'
                        for (x,y,w,h) in faces2:
                            cv2.rectangle(bigimg,(int(x*(1/aspect)),int((y)*(1/aspect))),(int((x+w)*(1/aspect)),int((y+h)*(1/aspect))),(255,0,0),2)
                    # print('They have {}. Cosine Similarity: {}, threshold: {} (higher value means higher similarity, max 1.0).'.format(msg, cosine_score, cosine_similarity_threshold))
                
                cv2.imshow("the movie",bigimg)
                k = cv2.waitKey(30) & 0xff
                if k==27:
                    break
            else:
                break
cv2.destroyAllWindows()