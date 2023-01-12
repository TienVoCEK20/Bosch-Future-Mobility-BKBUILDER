import socket
import struct
import time
import numpy as np

from threading import Thread

import cv2

from src.templates.workerprocess import WorkerProcess

class DetectionProcess(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        """Process used for sending images over the network to a targeted IP via UDP protocol 
        (no feedback required). The image is compressed before sending it. 

        Used for visualizing your raspicam images on remote PC.
        
        Parameters
        ----------
        inPs : list(Pipe) 
            List of input pipes, only the first pipe is used to transfer the captured frames. 
        outPs : list(Pipe) 
            List of output pipes (not used at the moment)
        """
        super(DetectionProcess,self).__init__( inPs, outPs)
        
    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads.
        """
        super(DetectionProcess,self).run()

    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the sending thread.
        """
        if self._blocker.is_set():
            return
        streamTh = Thread(name='DetectionThread',target = self._send_thread, args= (self.inPs[0], ))
        streamTh.daemon = True
        self.threads.append(streamTh)
      
    # ===================================== SEND THREAD ==================================
    def _send_thread(self, inP):
        """Sending the frames received thought the input pipe to remote client by using the created socket connection. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe to read the frames from CameraProcess or CameraSpooferProcess. 
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

        while True:
            try:
                stamps, image = inP.recv()
                #TODO
                

            except Exception as e:
                print("Detection failed to detect images:",e,"\n")
                pass