#!/bin/bash

set -e
cd /home/rishab/licenseplaterecognition/yolov5; python detector_api.py &
cd /home/rishab/licenseplaterecognition/deeptext; python deeptext_api.py &
cd /home/rishab/licenseplaterecognition; python api_v2.py