import json
import socket
import math
import time
import numpy as np
import cv2 as cv
from threading       import Thread

from src.templates.workerprocess import WorkerProcess

class CarControl(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        """Run on raspberry. It forwards the control messages received from socket to the serial handler
        
        Parameters
        ------------
        inPs : list(Pipe)
            List of input pipes (not used at the moment)
        outPs : list(Pipe) 
            List of output pipes (order does not matter)
        """

        super(CarControl,self).__init__( inPs, outPs)
        self.lane_result = inPs[0]
        self.object_result = inPs[1]
        self.control = outPs
        #print(self.control[0])
        self.activate = 1
        
        self.error_arr = np.zeros(5)
        self.time_pid = time.time()
        
        self.current_speed = 0
        self.pid_active = False
        self.activatePID()
    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads
        """
        super(CarControl,self).run()

    def activatePID(self, activate = True):
        self.pid_active = activate
        _activate = {
                    "action": "4",
                    "activate": self.pid_active
                }
        print(_activate) 
        self.control[0].send(_activate)
        
    def isPIDActive(self):
        return self.pid_active

    def checkInPs(self):
        print(self.lane_result)

    def updateSpeed(self, speed):
        self.current_speed = speed
        _speed =    {
                    "action": "1",
                    "speed": float(self.current_speed)
                }  
        print(_speed) 
        self.control[0].send(_speed)
        
    def adjustSpeed(self, value):
       self.updateSpeed(self.current_speed + value)
       
    def getCurrentSpeed(self):
        return self.current_speed
        
    def goForward(self, distance, speed):
        self.current_speed = speed
        _distance = {
                    "action": "7",
                    "distance": float(distance),
                    "speed": float(self.current_speed)
                }
        print(_distance) 
        self.control[0].send(_distance)
        
    def steerAngle(self, steerAngle):
        _steer = {
                    "action": "2",
                    "steerAngle": float(steerAngle)
                }
        print(_steer) 
        self.control[0].send(_steer)
       
    ''' Based on Mr. Ngo Duc Tuan's DR2020 code ''' 
    def steerPID(self, x, y):
        steer = 0
        if x == 0 and y == 0:
            steer = 0
        elif x <= 0:
            steer = -23
        elif x >= 360:
            steer = 23
        else:
            smooth_x = 5 * (x // 5)
            error = smooth_x - 180
            
            ''' Update error_arr '''
            self.error_arr[1:] = self.error_arr[0:-1]
            self.error_arr[0] = error
            
            ''' Get delta_t and update time_pid'''
            delta_t = time.time() - self.time_pid
            self.time_pid = time.time()
            
            ''' Initial p, i, d values given by BFMC source code '''
            kp = 0.115000
            ki = 0.810000
            kd = 0.000222
            
            ''' Calculate P (Proportional) '''
            P = error * kp
            
            ''' Calculate I (Integral) '''
            I = np.sum(self.error_arr) * delta_t * ki
            
            ''' Calculate D ( Derivative) '''
            D = (self.error_arr[0] - self.error_arr[1]) / delta_t * kd
            
            ''' Get steer value '''
            steer = P + I + D
            
            if math.isnan(steer):
                steer = 0
                
            if abs(steer) > 23:
                steer = np.sign(steer) * 23
        return int(steer)
    
    ''' Compute steering angle based on Vu Thanh Dat's DR2020 code '''
    def angleCalculator(self, x, y):
        angleDegree = 0

        if x != 0 or y != 0:
            slope = (x - 72) / float (y - 144) # (72, 144) is center of (144, 144) image
            angleRadian = float(math.atan(slope))
            angleDegree = float(angleRadian * 180.0 / math.pi)

        return angleDegree
    
    def computeCenter(self, roadImg):
        count = 0
        center_x = 0
        center_y = 0
        for i in range(0, 144):
            for j in range(0, 144):
                if roadImg[i][j] == 255:
                    count += 1
                    center_x += j
                    center_y += i
    

        if center_x != 0 or center_y != 0 or count != 0:
            center_x = center_x / count
            center_y = center_y / count

        return center_x, center_y
        
    def brake(self, brake_steerAngle):
        _brake = {  
                "action": "3",
                "brake (steerAngle)": float(brake_steerAngle)
            }
        print(_brake) 
        self.control[0].send(_brake)
#  # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the read thread to transmite the received messages to other processes. 
        """
        readTh = Thread(name='ReceiverCommandThread',target = self._read_stream, args = (self.outPs, ))
        self.threads.append(readTh)

    # ===================================== READ STREAM ==================================
    def _read_stream(self, outPs):
        """Receive the message and forwards them to the SerialHandlerProcess. 
        
        Parameters
        ----------
        outPs : list(Pipe)
            List of the output pipes.
        """
        try:
            while True:
                # print("hello")
                frame = self.lane_result.recv()[0]
                birdeye_img = frame['thresh']    
                birdeye_img = cv.resize(birdeye_img, (144,144))   
                print("sum of thresh: {}".format(np.sum(birdeye_img)))
                print(birdeye_img)            
                center_x, center_y = self.computeCenter(birdeye_img)
                print("center_x: {}\ncenter_y: {}".format(center_x, center_y))
                steer_angle = self.angleCalculator(center_x, center_y)
                print("Steering angle: {}".format(steer_angle))
                self.steerAngle(steer_angle)
                #self.steerAngle(10)
        except Exception as e:
            print(e)
            #2:0.00;;