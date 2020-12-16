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
    def quantStats(self, opt, audioFeatures, videoFeatures):
        #gets pairwise distances between video and audio
        dists = calc_pdist(videoFeatures, audioFeatures, vshift=opt.vshift)

        mdist = torch.mean(torch.stack(dists, 1), 1)
        #finds the min distance
        minval, minidx = torch.min(mdist, 0)
        #unclear
        offset = opt.vshift-minidx
        #confidence is median distance - min distance
        conf = torch.median(mdist) - minval
        fdist = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf = torch.median(mdist).numpy() - fdist
        fconfm = signal.medfilt(fconf, kernel_size=9)
        #median distance
        medianDistance = torch.median(mdist)
        self.median_distance.append(medianDistance)
        self.AV_offset.append(offset)
        self.min_distance.append(minval)
        self.confidence.append(conf)

        
    #aggregate class member lists and find averages and print them out 
    def aggregateQuantStats(self):
        print("placeholder")


if __name__ == '__main__':
    #run core of stats 
    intialTest = Stats()
    intialTest.processFeatures()
    intialTest.aggregateQuantStats()




