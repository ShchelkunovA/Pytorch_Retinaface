from .detector import Detector
# from . import *

def get_detector(weights='', cpu=False, thresh=0.9):
    return Detector(trained_model=weights, cpu=cpu, detector_thresh=thresh)