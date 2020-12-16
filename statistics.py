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
import matplotlib.pyplot as plt
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
        features_folder = '/datac/nkanama/facebookDataset/output_model_fake/pywork/features'
        for directory in os.listdir(features_folder):
            #cpu 
            audio_features = torch.load(os.path.join(features_folder,directory,'audioFeatures.pt'))
            video_features = torch.load(os.path.join(features_folder, directory, 'videoFeatures.pt'))
            self.quantStats(audio_features,video_features)

    #calculate median, min distance and confidence and AV offset for audio/video features
    #add to storage in class member variables
    def quantStats(self, audioFeatures, videoFeatures):
        #gets pairwise distances between video and audio
        vshift = 10
        dists = calc_pdist(videoFeatures, audioFeatures, vshift=10)

        mdist = torch.mean(torch.stack(dists, 1), 1)
        #finds the min distance
        minval, minidx = torch.min(mdist, 0)
        #standard offset - min distance
        offset = vshift-minidx
        #print("the offset is ", offset)
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
    def aggregateStats(self):
        print("average min distance ", sum(self.min_distance)/len(self.min_distance))
        print("average median distance ", sum(self.median_distance)/len(self.median_distance))
        print("average offset ", sum(self.AV_offset)//len(self.AV_offset))
        print("confidence ", sum(self.confidence)/len(self.confidence))

        #visualization of offset over time
        plt.plot(self.AV_offset)
        plt.xlabel("number of videos")
        plt.ylabel("audio-visual offset")
        plt.title("offset over 10 sec videos")
        stats_folder = '/datac/nkanama/facebookDataset/output_model_fake/pywork/features'
        plt.savefig(os.path.join(stats_folder, "fakeVideoOffset.png"))


if __name__ == '__main__':
    #run core of stats 
    intialTest = Stats()
    intialTest.processFeatures()
    intialTest.aggregateStats()




