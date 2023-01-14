import numpy as np
from .camera import Camera
from threading import Thread

from src.templates.workerprocess import WorkerProcess

class LaneDetectionProcess(WorkerProcess):

    def __init__(self, inPs, outPs):

        super(LaneDetectionProcess, self).__init__(inPs, outPs)
        self.outPs = outPs
        self.inPs = inPs
        self.camera = Camera()
        self.threads = list()

    def _init_threads(self):
        return super()._init_threads()

    def run(self):
        """
        Keys:
            birdeye_img: frame of bird eye view -> Shape (360, 640)
            inverse_transform: POV to convert into the original view
            transfored_view: POV to convert into bird view
            gray: gray scale of original image
            thresh: segmentation r.s.t the masking with threshold
            canny: edges

        """
        processing_result = self.camera.laneDetector.processor.process(self.inPs)
        return processing_result
