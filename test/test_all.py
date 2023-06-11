import pytest
import sys
import os
import cv2

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# from face_detect import idnetify_faces_image
# from face_detect import detect_face
import face_detect

@pytest.mark.parametrize("test_faces, base_image , result",[
    ("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg", "D:\\work\\github\\face-detect\\ID-Card1.jpg",[True,False]),
    ("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg", "D:\\work\\github\\face-detect\\ID-Card1.jpg1",[False])
])
# @pytest.mark.parametrize("parameters , result",[
#     ("-p D:\\work\\github\\face-detect\\ID-Card1.jpg -m D:\\work\\github\\face-detect\\VID_20150718_134620.mp4",[True,False])
#     # ("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg", "", "D:\\work\\github\\face-detect\\ID-Card1.jpg1",[False,False])
# ])
def test_idnetify_faces_image_new(test_faces, base_image, result):
# def test_idnetify_faces_image(parameters, result):
    show_detected = False
   
    # test_img = cv2.imread(test_faces.strip())
    # face_number,imgs_detected_test = face_detect.detect_face(test_img)

    # base_img = cv2.imread(base_image.strip())
    # _,img_detected_base = face_detect.detect_face(base_img)

    # sys.argv[1] = test_faces
    # sys.argv[2] = test_movie
    # sys.argv[3] = base_image

    assert face_detect.idnetify_faces_image_new(test_faces, base_image, show_detected) == result