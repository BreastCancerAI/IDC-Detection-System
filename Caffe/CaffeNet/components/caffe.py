############################################################################################
#
# The MIT License (MIT)
# 
# IDC Classifier CaffeNet Trainer
# Copyright (C) 2018 Adam Milton-Barker (AdamMiltonBarker.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#
# Title:         Caffe Helper
# Description:   Helper functions for Caffe.
# Configuration: data/confs.json
# Last Modified: 2018-09-04
############################################################################################

import cv2, json
from caffe.proto import caffe_pb2

class CaffeHelper():
    
    def __init__(self):

        self._confs = {}
        
        with open('data/confs.json') as confs:

            self._confs = json.loads(confs.read())
            
    def transform(self, img):
        
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        
        return cv2.resize(img,(self._confs["ClassifierSettings"]["imageWidth"],self._confs["ClassifierSettings"]["imageHeight"]),interpolation = cv2.INTER_CUBIC)

    def createDatum(self, imageData, label):
    
        datum = caffe_pb2.Datum()
        datum.channels = imageData.shape[2]
        datum.height = imageData.shape[0]
        datum.width = imageData.shape[1]
        datum.data = imageData.tobytes()
        datum.label = int(label)

        return datum