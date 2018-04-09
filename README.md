# OpenCV_DepthMapMirror

To run the code, one must first have a few essentials. 

The first items that you need
are two web cams of equal quality/FOV that must be placed on the same horizontal plane.

Second, you must have a checker board, either real or printed out to calibrate the 
cameras. 

Once you have those, you need to run Capture.py for about 20 seconds or more to
get a decent number of photos for calibration. Next, you need to run Cali.py with
3 arguments, the first two being the folders of the left and right images, respectively. 
The third argument needs to be the name of the output file. After that, a model should 
now be made and can be used to see the depth map by running DepthMap.py with the 
output file form Cali.py as its argument.