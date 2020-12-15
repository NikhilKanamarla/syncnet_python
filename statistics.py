import torch
import numpy
import time
import pdb
import argparse
import subprocess
import os
import math
import glob
import sys
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree
from SyncNetInstance import calc_pdist

class Stats:
    def __init__(self, median_distance = None, min_distance = None, confidence = None, AV_offset=None ):
        self.median_distance = []
        self.min_distance = []
        self.confidence = []
        self.AV_offset = []
    #iterate through folder and load video/audio features
    #send to the quantative stats method for processing
    def processFeatures(self):
        print("placeholder")
    #calculate median, min distance and confidence and AV offset for audio/video features
    #add to storage in class member variables
    def quantStats(self, audioFeatures, videoFeatures):
        print("placeholder")
    #aggregate class member lists and find averages and print them out 
    def aggregateQuantStats(self):
        print("placeholder")


if __name__ == '__main__':
    #run core of stats 
    intialTest = Stats()
    intialTest.processFeatures()
    intialTest.aggregateQuantStats()




