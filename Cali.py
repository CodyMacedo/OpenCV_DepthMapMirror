import cv2
import numpy as np
import glob
import os
import random
import sys



criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30,
        0.001)

        
chessboard_size = (7, 6)       
chessboard = np.zeros((chessboard_size[0] * chessboard_size[1], 3),
        np.float32)
chessboard[:, :2] = np.mgrid[0:chessboard_size[0],
        0:chessboard_size[1]].T.reshape(-1, 2)

leftDir = sys.argv[1]
rightDir = sys.argv[2]
outputFile = sys.argv[3]

def readImages(dir):
    cacheFile = "{0}/chessboards.npz".format(dir)
    try:
        cache = np.load(cacheFile)
        return (list(cache["filenames"]), list(cache["objectPoints"]),
                list(cache["imagePoints"]), tuple(cache["imageSize"]))
    except IOError:
        print("Cache file at {0} not found".format(cacheFile))
    imagePaths = glob.glob("{0}/*.jpg".format(dir))

    
    filenames = []
    objectPoints = []
    imagePoints = []
    imageSize = None

    
    for imagePath in sorted(imagePaths):
        img = cv2.imread(imagePath)
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imageSize = grayImg.shape[::-1]

        hasCorners, corners = cv2.findChessboardCorners(grayImg,
                chessboard_size, cv2.CALIB_CB_FAST_CHECK)

        if hasCorners:
            filenames.append(os.path.basename(imagePath))
            objectPoints.append(chessboard)
            cv2.cornerSubPix(grayImg, corners, (11, 11), (-1, -1),
                    criteria)
            imagePoints.append(corners)

        cv2.drawChessboardCorners(img, chessboard_size, corners, hasCorners)
        cv2.imshow(dir, img)
        cv2.waitKey(1)

    cv2.destroyWindow(dir)

    # Save a cached version of the calibration so we don't have to calculate it every time
    np.savez_compressed(cacheFile,
            filenames=filenames, objectPoints=objectPoints,
            imagePoints=imagePoints, imageSize=imageSize)
            
            
    return filenames, objectPoints, imagePoints, imageSize

    
    
    
(leftFilenames, leftObjectPoints, leftImagePoints, leftSize
        ) = readImages(leftDir)
(rightFilenames, rightObjectPoints, rightImagePoints, rightSize
        ) = readImages(rightDir)

if leftSize != rightSize:
    print("Camera resolutions do not match")
    sys.exit(1)
imageSize = leftSize

filenames = list(set(leftFilenames) & set(rightFilenames))
print(len(filenames))
if (len(filenames) > 64):
    print("Using {0} randomly selected images to calibrate"
            .format(64))
    filenames = random.sample(filenames, 64)
filenames = sorted(filenames)


def getMaches(requestedFilenames,
        allFilenames, objectPoints, imagePoints):
    requestedFilenameSet = set(requestedFilenames)
    requestedObjectPoints = []
    requestedImagePoints = []

    for index, filename in enumerate(allFilenames):
        if filename in requestedFilenameSet:
            requestedObjectPoints.append(objectPoints[index])
            requestedImagePoints.append(imagePoints[index])

    return requestedObjectPoints, requestedImagePoints

leftObjectPoints, leftImagePoints = getMaches(filenames,
        leftFilenames, leftObjectPoints, leftImagePoints)
rightObjectPoints, rightImagePoints = getMaches(filenames,
        rightFilenames, rightObjectPoints, rightImagePoints)

objectPoints = leftObjectPoints


_, leftCameraMatrix, leftDistortionCoefficients, _, _ = cv2.calibrateCamera(
        objectPoints, leftImagePoints, imageSize, None, None)

_, rightCameraMatrix, rightDistortionCoefficients, _, _ = cv2.calibrateCamera(
        objectPoints, rightImagePoints, imageSize, None, None)

# Calibrate the cameras together
(_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        objectPoints, leftImagePoints, rightImagePoints,
        leftCameraMatrix, leftDistortionCoefficients,
        rightCameraMatrix, rightDistortionCoefficients,
        imageSize, None, None, None, None,
        cv2.CALIB_FIX_INTRINSIC, criteria)

# Rectifying the cameras
(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                leftCameraMatrix, leftDistortionCoefficients,
                rightCameraMatrix, rightDistortionCoefficients,
                imageSize, rotationMatrix, translationVector,
                None, None, None, None, None,
                cv2.CALIB_ZERO_DISPARITY, 0.25)

# Save a cached version of the calibration
leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        leftCameraMatrix, leftDistortionCoefficients, leftRectification,
        leftProjection, imageSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        rightCameraMatrix, rightDistortionCoefficients, rightRectification,
        rightProjection, imageSize, cv2.CV_32FC1)

np.savez_compressed(outputFile, imageSize=imageSize,
        leftMapX=leftMapX, leftMapY=leftMapY, leftROI=leftROI,
        rightMapX=rightMapX, rightMapY=rightMapY, rightROI=rightROI)

cv2.destroyAllWindows()