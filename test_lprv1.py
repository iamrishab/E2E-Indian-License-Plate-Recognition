#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'

import os
import cv2
import numpy as np

import config
from LPRv1.detect_license_plate import CarNumberPlateDetectorv1
from LPRv1.recognize_text_from_license_plate import TesserectRecognizeText


def main():
    lprv1 = CarNumberPlateDetectorv1()
    trt = TesserectRecognizeText()
    
     # reading camera feed
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if type(config.VIDEO_SOURCE) != str:
        # setting camera parameters
        cap.set(config.FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH)
        cap.set(config.FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT)

    # reading feed from camera
    while cap.isOpened():
        ret, frame_bgr = cap.read()

        # close if the input source cannot fetch any frame
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)
        bbs = lprv1.detect(frame_rgb)
        result = {}

        if len(bbs) == 1:
            xmin, ymin, xmax, ymax = bbs[0]
            cropped_image = frame_bgr[ymin:ymax, xmin:xmax, :]
            text = trt.recognize_text(cropped_image)
            result = {'response': {f'{text}': [xmin, ymin, xmax, ymax]}}
            print(result)

    cv2.destroyAllWindows()
    cap.release()

    
if __name__ == '__main__':
    main()