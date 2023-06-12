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
    ("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg", "D:\\work\\github\\face-detect\\ID-Card1.jpg1",[False]),
    ("", "D:\\work\\github\\face-detect\\ID-Card1.jpg1",[False]),
    ("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg", "",[False]),
    ("", "",[False])
])
def test_idnetify_faces_image_new(test_faces, base_image, result):
    show_detected = False
    assert face_detect.idnetify_faces_image_new(test_faces, base_image, show_detected) == result
    print("every things have done")

@pytest.mark.parametrize("base_image , result",[
    ("D:\\work\\github\\face-detect\\IMG_20150925_153809.jpg",2),
    ("D:\\work\\github\\face-detect\\ID-Card1.jpg",1),
    ("",0),
    (" ",0),
])
def test_detect_face(base_image, result):
    base_img = cv2.imread(base_image.strip())
    i,detected = face_detect.detect_face(base_img)
    assert i == result 
    assert len(detected) == result
    for i in range(len(detected)):
        assert len(detected[i]) > 50
    print("every things have done")
