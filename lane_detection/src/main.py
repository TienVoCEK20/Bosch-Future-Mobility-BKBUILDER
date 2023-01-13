import cv2 as cv
from laneDetection import Preprocessor
from utils import *
import glob
from camera import Camera
import matplotlib.pyplot as plt

NUM_FRAMES = 500 
OFFSET = 10

def draw_points(img, points):
    
    try:
        for idx in range(4):
            cv.circle(img, (
                (int(points[idx][0])), int(points[idx][1])
            ), 15, (0,0,255), cv.FILLED)    
        return img
    except Exception as e:
        print(e)

if __name__ == '__main__':

    # extract_frames(VIDEO_PATH, NUM_FRAMES, OFFSET)

    preprocessor = Preprocessor()
    video = cv.VideoCapture(VIDEO_PATH)
    camera = Camera()
    camera.load_calibrate_info(CALIBRATE_PICKLE)
    

    while True:
        try:
            flag, frame = video.read()
            frame = cv.resize(frame, IMG_SIZE)
            calibrate_img = camera.undistort(frame)
            output1 = camera.laneDetector.processor.process(calibrate_img)
            output = camera.detectLane(calibrate_img)

            


            # plt.imshow(frame, cmap= 'gray')
            # plt.show()
            # print(output1['birdeye'])
            output = draw_points(output, output1['birdeye']['src'])
            # cv.imshow('points', points)
            cv.imshow("Frame", output1['thresh'])
            cv.imshow("Lane detection", output) 
            cv.waitKey(1)
        
        except Exception as e:
            print(e)
   

