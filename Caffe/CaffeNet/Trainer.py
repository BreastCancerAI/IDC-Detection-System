############################################################################################
#
# The MIT License (MIT)
# 
# IDC Classifier CaffeNet DataSorter
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
# Title:         IDC Classifier CaffeNet DataSorter
# Description:   Sorts the data for the IDC Classifier.
# Configuration: data/confs.json
# Last Modified: 2018-08-09
#
# Example Usage:
#
#   $ python3.5 Train.py
#
############################################################################################

import os, glob, random, cv2, caffe, lmdb, json

import numpy as np

from caffe.proto import caffe_pb2
from components.caffe import CaffeHelper

class DataSorter():
    
    def __init__(self):
                
        self._confs = {}
        self.trainData = []
        self.classNames = []
        self.trainSize = 0
        self.validationSize = 0
        self.testSize = 0
        self.labels = None
        self.trainLMDB = None

        self.CaffeHelper = CaffeHelper()
        
        with open('data/confs.json') as confs:
            self._confs = json.loads(confs.read())
        
        os.system('rm -rf  ' + self._confs["ClassifierSettings"]["trainLMDB"])
        os.system('rm -rf  ' + self._confs["ClassifierSettings"]["validationLMDB"])

    def sortData(self):

        self.labels = open(self._confs["ClassifierSettings"]["labels"],"w")
        self.labels.write("classes\n")

        for dirName in os.listdir(self._confs["ClassifierSettings"]["dataset_dir"]):

            path = os.path.join(self._confs["ClassifierSettings"]["dataset_dir"], dirName)

            if os.path.isdir(path):

                self.classNames.append(path)
                self.labels.write(dirName+"\n")
        
        self.labels.close()

        for directory in self.classNames:

            for filename in os.listdir(directory):

                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.gif'):

                    path = os.path.join(directory, filename)
                    self.trainData.append(path)

                else:

                    continue

    def createLMDB(self):

        print("CREATING LMDB")

        self.trainSize = int((len(self.trainData) * self._confs["ClassifierSettings"]["trainCut"]) / 100)
        self.validationSize = int((len(self.trainData) * self._confs["ClassifierSettings"]["validationCut"]) / 100)
        self.testSize = int((len(self.trainData) * self._confs["ClassifierSettings"]["testCut"]) / 100)
        
        self.trainingData = self.trainData[:self.trainSize]
        self.validationData = self.trainData[:self.validationSize]
        self.testData = self.trainData[:self.testSize]

        print("")
        print("Data Size:       "+str(len(self.trainData)))
        print("Training Size:   "+str(len(self.trainingData)))
        print("Validation Size: "+str(len(self.validationData)))
        print("Test Size:       "+str(len(self.testData)))
        print("")
        print("CREATING TRAINING LMDB ")
        print("PLEASE WAIT THIS MAY TAKE A WHILE ")
        print("")

        random.shuffle(self.trainingData)
        
        trainer = lmdb.open(
            self._confs["ClassifierSettings"]["trainLMDB"], 
            map_size=int(1e12))

        with trainer.begin(write=True) as i:

            count = 0
            for data in self.trainingData:

                label = os.path.basename(os.path.dirname(data))

                i.put(
                    '{:08}'.format(count).encode('ascii'), 
                    self.CaffeHelper.createDatum(
                        cv2.resize(
                            self.CaffeHelper.transform(
                                cv2.imread(data, cv2.IMREAD_COLOR)
                            ), 
                            (self._confs["ClassifierSettings"]["imageHeight"], self._confs["ClassifierSettings"]["imageWidth"])), 
                        label
                    ).SerializeToString())
                    
                count = count + 1

        trainer.close()

        print("DATA COUNT: "+str(count))
        print("")
        print("CREATING VALIDATION LMDB ")
        print("PLEASE WAIT THIS MAY TAKE A WHILE ")
        print("")

        random.shuffle(self.validationData)
        
        validator = lmdb.open(
            self._confs["ClassifierSettings"]["validationLMDB"], 
            map_size=int(1e12))

        with validator.begin(write=True) as i:

            count = 0
            for data in self.validationData:

                label = os.path.basename(os.path.dirname(data))

                i.put(
                    '{:08}'.format(count).encode('ascii'), 
                    self.CaffeHelper.createDatum(
                        cv2.resize(
                            self.CaffeHelper.transform(
                                cv2.imread(data, cv2.IMREAD_COLOR)
                            ), 
                            (self._confs["ClassifierSettings"]["imageHeight"], self._confs["ClassifierSettings"]["imageWidth"])), 
                        label
                    ).SerializeToString())
                    
                count = count + 1

        validator.close()

        print("DATA COUNT: "+str(count))

    def computeMean(self):

        print("COMPUTING MEAN")

        os.system('compute_image_mean -backend=lmdb  ' + self._confs["ClassifierSettings"]["trainLMDB"] + ' ' + self._confs["ClassifierSettings"]["proto"])

        
DataSorter = DataSorter()
DataSorter.sortData()
DataSorter.createLMDB()
DataSorter.computeMean()