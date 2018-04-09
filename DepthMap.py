import cv2
import numpy as np
import sys


REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048

if len(sys.argv) != 2:
    print("Syntax: {0} CALIBRATION_FILE".format(sys.argv[0]))
    sys.exit(1)

calibration = np.load(sys.argv[1], allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

WIDTH_RES = 1280

# TODO: Use more stable identifiers
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


stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setROI1(leftROI)
stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    
    _, leftImg = leftCam.read()
    leftImg = crop_img(leftImg)
    leftHeight, leftWidth = leftImg.shape[:2]
    _, rightImg = rightCam.read()
    rightImg = crop_img(rightImg)
    rightHeight, rightWidth = rightImg.shape[:2]

    if (leftWidth, leftHeight) != imageSize:
        print("Left camera has different size than the calibration data")
        break

    if (rightWidth, rightHeight) != imageSize:
        print("Right camera has different size than the calibration data")
        break

    fixedLeft = cv2.remap(leftImg, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(rightImg, rightMapX, rightMapY, REMAP_INTERPOLATION)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    cv2.imshow('left', fixedLeft)
    cv2.imshow('right', fixedRight)
    cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

leftCam.release()
rightCam.release()
cv2.destroyAllWindows()