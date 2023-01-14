import sys
sys.path.append('.')

import time
import socket
import struct
import numpy as np


import cv2
from threading import Thread

import multiprocessing
from multiprocessing import Process,Event

from src.templates.workerprocess import WorkerProcess

from src.data.vehicletovehicle.vehicletovehicle import vehicletovehicle

class ServerReceiverProcess(WorkerProcess):
    def __init__(self, inPs, outPs):
        '''
        Document: TBD
        '''
        super(ServerReceiverProcess, self).__init__(inPs, outPs)
        
    def run(self):
        super(ServerReceiverProcess, self).run()
    
    def _init_threads(self):
        print("come here")
        v2vTh = vehicletovehicle()
        self.threads.append(v2vTh)

    def _send_thread(self, inP):
        pass