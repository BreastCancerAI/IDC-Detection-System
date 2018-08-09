############################################################################################
#
# The MIT License (MIT)
# 
# Facenet Helpers
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
# Title:         Facenet Helpers
# Description:   Helper functions for Facenet.
# Configuration: data/confs.json
# Last Modified: 2018-08-09
#
############################################################################################

import os, json, cv2
import numpy as np
from datetime import datetime

from tools.OpenCV import OpenCVHelpers as OpenCVHelpers

class FacenetHelpers():
    
    def __init__(self):
        
        self.OpenCVHelpers = OpenCVHelpers()
        
    def infer(self, image_to_classify, facenet_graph):
        
        # get a resized version of the image that is the dimensions
        # SSD Mobile net expects
        resized_image = self.preprocess(image_to_classify)

        #cv2.imshow("preprocessed", resized_image)

        # ***************************************************************
        # Send the image to the NCS
        # ***************************************************************
        facenet_graph.LoadTensor(resized_image.astype(np.float16), None)

        # ***************************************************************
        # Get the result from the NCS
        # ***************************************************************
        output, userobj = facenet_graph.GetResult()

        #print("Total results: " + str(len(output)))
        #print(output)

        return output

    def match(self, face1_output, face2_output):
        if (len(face1_output) != len(face2_output)):
            print('-- Length mismatch in match')
            return False
        total_diff = 0
        for output_index in range(0, len(face1_output)):
            this_diff = np.square(face1_output[output_index] - face2_output[output_index])
            total_diff += this_diff
        print('-- Total Difference is: ' + str(total_diff))

        if (total_diff < 1.3):
            # the total difference between the two is under the threshold so
            # the faces match.
            return "True", total_diff
        else:
            return "False", total_diff


    # create a preprocessed image from the source image that matches the
    # network expectations and return it
    def preprocess(self, src):
        # scale the image
        NETWORK_WIDTH = 160
        NETWORK_HEIGHT = 160
        preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

        #convert to RGB
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

        #whiten
        preprocessed_image = self.OpenCVHelpers.whiten(preprocessed_image)

        # return the preprocessed image
        return preprocessed_image