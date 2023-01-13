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
        #print(self.control[0])
        self.activate = 1
        
        self.error_arr = np.zeros(5)
        self.time_pid = time.time()
    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads
        """
        super(CarControl,self).run()
    def activatePID(self):
        _activate_pid = {
                    "action": "4",
                    "activate": True
                }
        command = _activate_pid
        print(command) 
        self.control[0].send(command)

    def keepGoing(self, speed):
        _speed =    {
                    "action": "1",
                    "speed": float(speed)
                }  
        command = json.loads(_speed)
        print(command) 
        self.control[0].send(command)
        
    def goForward(self, distance, speed):
        _distance = {
                    "action": "7",
                    "distance": float(distance),
                    "speed": float(speed)
                }
        command = _distance
        print(command) 
        self.control[0].send(command)
        
    def steerAngle(self, steerAngle):
        _distance = {
                    "action": "2",
                    "steerAngle": float(steerAngle)
                }
        command = json.loads(_distance)
        print(command) 
        self.control[0].send(command)
       
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
    
    ''' Compute steering angle based on Vu Thanh Dat's DR2020 code '''
    def angleCalculator(self, x, y):
        slope = (x - 72) / float(y - 144) # (320, 360) is center of (640, 360) image
        angleRadian = float(math.atan(slope))
        angleDegree = float(angleRadian * 180.0 / math.pi)
        return angleDegree
    
    def computeCenter(self, roadImg):
        count = 0
        center_x = 0
        center_y = 0 
        
        for i in range(0,144):
            for j in range(0,144):
                if roadImg[i][j] == 255:
                    count += 1
                    center_x += i
                    center_y += j
    

        if center_x != 0 or center_y != 0 or count != 0:
            center_x = center_x / count
            center_x = center_y / count
            angleDegree = self.angleCalculator(center_x, center_y) # Call angle_calculator method in speed_up.py to use numba function

        return angleDegree
        
    def brake(self, break_steerAngle):
        _brake = {  
                "action": "3",
                "brake (steerAngle)": float(break_steerAngle)
            }
        command = json.loads(_brake)
        print(command) 
        self.control[0].send(command)
 # ===================================== INIT THREADS =================================
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
                #print("hello")
                if self.activate: 
                    self.activate = 0
                    self.activatePID()
                    self.goForward(1,0.09)

        except Exception as e:
            print(e)