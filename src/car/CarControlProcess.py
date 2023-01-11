import json
import socket

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
    def brake(self, break_steerAngle):
        _brake = {  
                "action": "3",
                "brake (steerAngle)": float(break_steerAngle)
            }
        command = json.dumps(_brake).encode()
        print(command) 
        self.control.send(command)

        