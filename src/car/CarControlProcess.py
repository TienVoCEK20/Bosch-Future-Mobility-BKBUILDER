import json
import socket
import math
import time
import numpy as np

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
        self.control = outPs
        
        self.error_arr = np.zeros(5)
        self.time_pid = time.time()
    
    def keepGoing(self, speed):
        _speed =    {
                    "action": "1",
                    "speed": float(speed)
                }  
        command = json.dumps(_speed).encode()
        print(command) 
        self.control.send(command)
        
    def goForward(self, distance, speed):
        _distance = {
                    "action": "7",
                    "distance": float(distance),
                    "speed": float(speed)
                }
        command = json.dumps(_distance).encode()
        print(command) 
        self.control.send(command)
        
    def steerAngle(self, steerAngle):
        _distance = {
                    "action": "2",
                    "steerAngle": float(steerAngle)
                }
        command = json.dumps(_distance).encode()
        print(command) 
        self.control.send(command)
       
    ''' Based on Mr. Ngo Duc Tuan's DR2020 code ''' 
    def steerPID(self, x, y):
        steer = 0
        if x == 0 and y == 0:
            steer = 0
        elif x <= 0:
            steer = -60
        elif x >= 360:
            steer = 60
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
                
            if abs(steer) > 60:
                steer = np.sign(steer) * 60
        return int(steer)
            
        
    def brake(self, break_steerAngle):
        _brake = {  
                "action": "3",
                "brake (steerAngle)": float(break_steerAngle)
            }
        command = json.dumps(_brake).encode()
        print(command) 
        self.control.send(command)

        