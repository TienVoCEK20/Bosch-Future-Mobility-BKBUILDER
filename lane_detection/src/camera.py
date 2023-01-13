import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils import SAVE_DIR, IMG_DIR, save_pkl, load_pkl
from laneDetection import LaneDetection

class Camera():    
    def __init__(self):
        # Stores the source 
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        
        self.objpoints = [] # 3D points in real space
        self.imgpoints = [] # 2D points in img space
    
        self.laneDetector =  LaneDetection()

    def calibrate_camera(self, imgList, nx = 9, ny = 6):
        if self.mtx == None:
            objp = np.zeros((ny*nx, 3), np.float32)
            objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

            for idx, img_file in enumerate(imgList):
                img = cv.imread(img_file, 1)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, (nx, ny), flags=cv.CALIB_CB_FAST_CHECK)
                
                if ret == True:
                    self.objpoints.append(objp)
                    self.imgpoints.append(corners)
            print(self.objpoints, self.imgpoints)
            results = dict()
            results['ret'], results['mtx'], results['dist'], results['rvecs'], results['tvecs'] = \
                                cv.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            self.ret = results['ret']
            self.mtx = results['mtx']
            self.dist = results['dist']
            self.rvecs = results['rvecs']
            self.tvecs = results['tvecs']

            print("Saving calibration...")
            save_pkl(results, SAVE_DIR+'/calibration.pkl')
            print("Saved!")
            return self.mtx, self.dist

    def load_calibrate_info(self, state_dict: str):
        try:
            results = load_pkl(state_dict)
            self.ret = results['ret']
            self.mtx = results['mtx']
            self.dist = results['dist']
            self.rvecs = results['rvecs']
            self.tvecs = results['tvecs']

            assert self.ret != None, "Failed to load calibrate information"
        except Exception as e:
            print(e)

    def undistort(self, img):
        return cv.undistort(img,self.mtx,self.dist,None,self.mtx)
    
    def detectLane(self, img):
        preprocess_reulsts = self.laneDetector.processor.process(img)
        thresh = preprocess_reulsts['thresh']
        inverse_transform = preprocess_reulsts['inverse_transform']

        hist, _, _ = self.laneDetector.plotHistogram(thresh)

        ploty, left_fit, right_fit, left_fitx, right_fitx = self.laneDetector.slide_window_search(thresh, hist)
        draw_info = self.laneDetector.general_search(thresh, left_fit, right_fit)
        curveRadius, curveDirection = self.laneDetector.measure_lane_curvature(ploty, left_fitx, right_fitx)
        
        meanPts, result = self.laneDetector.draw_lane_lines(img, thresh, inverse_transform, draw_info)


        deviation, directionDev = self.laneDetector.offCenter(meanPts, img)


        # Adding text to our final image
        finalImg = self.laneDetector.addText(result, curveRadius, curveDirection, deviation, directionDev)
        print(finalImg.shape)
        return finalImg