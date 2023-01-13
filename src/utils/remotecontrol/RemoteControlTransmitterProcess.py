# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import json
import socket
import time

from threading       import  Thread
from multiprocessing import  Pipe

from src.utils.remotecontrol.RcBrainThread              import RcBrainThread
from src.utils.remotecontrol.KeyboardListenerThread     import KeyboardListenerThread
from src.templates.workerprocess                        import WorkerProcess

activate_pid = {
    "action": "4",
    "activate": True
}
speed = {
  "action": "1",
  "speed": float(0.09)
}

brake = {  
    "action": "3",
    "brake (steerAngle)": float(20)
}

encoder = {
    "action": "5",
    "activate": True
}

# pid_control = {
#     "action": "6",
#     "kp"      = float(1),
#     "ki"      = float(1),
#     "kd"      = float(1),
#     "tf"      = float(1)
# }

distance = {
    "action": "7",
    "distance": float(1),
    "speed": float(0.09)
}
class RemoteControlTransmitterProcess(Thread):
    # ===================================== INIT==========================================
    def __init__(self,  inPs = [], outPs = []):
        """Run on the PC. It forwards the commans from the user via KeboardListenerThread to the RcBrainThread. 
        The RcBrainThread converts them into actual commands and sends them to the remote via a socket connection.
        
        """
        super(RemoteControlTransmitterProcess,self).__init__()

        # Can be change to a multithread.Queue.
        self.lisBrR, self.lisBrS = Pipe(duplex=False)

        self.rcBrain   =  RcBrainThread()
        self.listener  =  KeyboardListenerThread([self.lisBrS])

        self.port      =  12244
        self.serverIp  = '192.168.1.8'

        self.threads = list()

        self.runOnce = 1
    # ===================================== RUN ==========================================
    def run(self):
        """Apply initializing methods and start the threads. 
        """
        self._init_threads()
        self._init_socket()
        for th in self.threads:
            th.start()

        for th in self.threads:
            th.join()

        super(RemoteControlTransmitterProcess,self).run()

    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the command sender thread for transmite the receiver process all commands. 
        """
        self.listener.daemon = self.daemon
        self.threads.append(self.listener)
        
        sendTh = Thread(name = 'SendCommandThread',target = self._send_command_thread, args=(self.lisBrR, ),daemon=self.daemon)
        self.threads.append(sendTh)

    # ===================================== INIT SOCKET ==================================
    def _init_socket(self):
        """Initialize the communication socket client.
        """
        self.client_socket = socket.socket(
                                family  = socket.AF_INET, 
                                type    = socket.SOCK_DGRAM
                            )
    def keepGoing(self, speed):
        _speed =    {
                    "action": "1",
                    "speed": float(0.09)
                }  
        command = json.dumps(_speed).encode()
        print(command) 
        self.client_socket.sendto(command,(self.serverIp,self.port))      
    def goForward(self, distance, speed):
        _distance = {
                    "action": "7",
                    "distance": float(distance),
                    "speed": float(speed)
                }
        command = json.dumps(_distance).encode()
        print(command) 
        self.client_socket.sendto(command,(self.serverIp,self.port))
    def steerAngle(self, steerAngle):
        _distance = {
                    "action": "2",
                    "steerAngle": float(steerAngle)
                }
        command = json.dumps(_distance).encode()
        print(command) 
        self.client_socket.sendto(command,(self.serverIp,self.port))
    def brake(self, break_steerAngle):
        _brake = {  
                "action": "3",
                "brake (steerAngle)": float(break_steerAngle)
            }
        command = json.dumps(_brake).encode()
        print(command) 
        self.client_socket.sendto(command,(self.serverIp,self.port))
        
    # ===================================== SEND COMMAND =================================
    def _send_command_thread(self, inP):
        """Transmite the command to the remotecontrol receiver. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe. 
        """
        while True:
        #     key = inP.recv()
        #     print('key: '+ key)
        #     command = self.rcBrain.getMessage(key)
        #     if command is not None:
        #         command = json.dumps(command).encode()
        #         print(command)
        #         self.client_socket.sendto(command,(self.serverIp,self.port))
            if self.runOnce:
                self.runOnce = 0
                command = json.dumps(activate_pid).encode()
                print(command) 
                self.client_socket.sendto(command,(self.serverIp,self.port))
                # command = json.dumps(speed).encode()
                # print(command) 
                # self.client_socket.sendto(command,(self.serverIp,self.port))
                # time.sleep(3)
                # self.brake(10)
                self.keepGoing(0.09)
                self.steerAngle(20)
                time.sleep(3)
                self.steerAngle(-20)
                time.sleep(3)
                self.brake(10)
                # command = json.dumps(encoder).encode()
                # print(command) 
                # self.client_socket.sendto(command,(self.serverIp,self.port))
                
                # time.sleep(3)

                # command = json.dumps(brake).encode()
                # print(command) 
                # self.client_socket.sendto(command,(self.serverIp,self.port))

                # command = json.dumps(distance).encode()
                # print(command) 
                # self.client_socket.sendto(command,(self.serverIp,self.port))
                #self.goForward(0.5,0.09)
