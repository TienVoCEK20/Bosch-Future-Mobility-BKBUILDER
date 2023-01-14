# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC orginazers
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#========================================================================
# SCRIPT USED FOR WIRING ALL COMPONENTS
#========================================================================
import sys
sys.path.append('.')

import time
import signal
from multiprocessing import Pipe, Process, Event 

# car imports
from src.car.CarControlProcess                              import CarControl
# hardware imports
from src.hardware.camera.cameraprocess                      import CameraProcess
from src.hardware.camera.CameraSpooferProcess               import CameraSpooferProcess
from src.hardware.serialhandler.SerialHandlerProcess        import SerialHandlerProcess

# utility imports
from src.utils.camerastreamer.CameraStreamerProcess         import CameraStreamerProcess
from src.utils.remotecontrol.RemoteControlReceiverProcess   import RemoteControlReceiverProcess

# detection imports
from src.utils.detection.DetectionProcess                   import DetectionProcess

#server imports
from src.utils.server.ServerReceiverProcess                 import ServerReceiverProcess
# =============================== CONFIG =================================================
enableStream        =  True
enableCameraSpoof   =  True 
enableRc            =  False
enableServer        =  True
# =============================== INITIALIZING PROCESSES =================================
allProcesses = list()

# =============================== HARDWARE ===============================================
if enableStream:
    camStR, camStS = Pipe(duplex = True)           # camera  ->  streamer
    camDtR, camDtS = Pipe(duplex= False)
    if enableCameraSpoof:
        camSpoofer = CameraSpooferProcess([],[camStS],videoDir='/home/pi/Bosch-Future-Mobility-BKBUILDER/testvideo/',ext='.avi')
        allProcesses.append(camSpoofer)

    else:
        camProc = CameraProcess([],[camStS])
        allProcesses.append(camProc)
    #DETECTION
    detectionProc = DetectionProcess([camStR], [camDtS])
    allProcesses.append(detectionProc)
    #STREAM
    streamProc = CameraStreamerProcess([camDtR], [])
    allProcesses.append(streamProc)

# =============================== SERVER ===================================================
if enableServer:
    SerStR, SerStS = Pipe(duplex = False)
    serverProc = ServerReceiverProcess([SerStR], [SerStS])
    allProcesses.append(serverProc)


# =============================== DATA ===================================================
#LocSys client process
# LocStR, LocStS = Pipe(duplex = False)           # LocSys  ->  brain
# from data.localisationsystem.locsys import LocalisationSystemProcess
# LocSysProc = LocalisationSystemProcess([], [LocStS])
# allProcesses.append(LocSysProc)

global car

# =============================== CONTROL =================================================
if enableRc:
    rcShR, rcShS   = Pipe(duplex = False)           # rc      ->  serial handler
    # serial handler process
    shProc = SerialHandlerProcess([rcShR], [])
    #print(rcShR)
    allProcesses.append(shProc)
    #print(rcShR)
    #rcProc = RemoteControlReceiverProcess([],[rcShS])
    rcProc = CarControl([],[rcShS])
    car = rcProc
    allProcesses.append(rcProc)

# =============================== DETECTION =================================================




# ===================================== START PROCESSES ==================================
print("Starting the processes!",allProcesses)
for proc in allProcesses:
    proc.daemon = True
    proc.start()

# count = 0
# car.activatePID()
# print(car.isPIDActive()) 
# car.updateSpeed(0.09)
# time.sleep(1)
# while car.getCurrentSpeed() < 0.2:
#     car.adjustSpeed(0.01)
#     print(car.getCurrentSpeed())
#     time.sleep(0.5)
# car.brake(0)

# ===================================== STAYING ALIVE ====================================
blocker = Event()  

try:
    blocker.wait()
except KeyboardInterrupt:
    print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")
    for proc in allProcesses:
        if hasattr(proc,'stop') and callable(getattr(proc,'stop')):
            print("Process with stop",proc)
            proc.stop()
            proc.join()
        else:
            print("Process witouth stop",proc)
            proc.terminate()
            proc.join()
