import cv2
import numpy as np

LEFT = "imgs/left/{:06d}.jpg"
RIGHT = "imgs/right/{:06d}.jpg"
fId = 0

WIDTH_RES = 1280

leftCam = cv2.VideoCapture(1)
rightCam = cv2.VideoCapture(2)

leftCam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
leftCam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
rightCam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
rightCam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Use MJPEG to avoid overloading the USB 2.0 bus at this resolution
leftCam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
rightCam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


def crop_img(img):
    return img[:,int((WIDTH_RES-960)/2):int(960+(WIDTH_RES-960)/2)]


while(True):

    _, leftImg = leftCam.read()
    leftImg = crop_img(leftImg)
    
    _, rightImg = rightCam.read()
    rightImg = crop_img(rightImg)
    
    
    cv2.imwrite(LEFT.format(fId), leftImg)
    cv2.imwrite(RIGHT.format(fId), rightImg)
    fId += 1
    
    cv2.imshow('left', leftImg)
    cv2.imshow('right', rightImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # breaks when the "q" key is pressed
        break
        
    
        
leftCam.release()
rightCam.release()
cv2.destroyAllWindows()