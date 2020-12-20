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
    def __init__(self, median_distance = None, min_distance = None, confidence = None, AV_offset=None, mean_distance=None):
        self.median_distance = []
        self.min_distance = []
        self.confidence = []
        self.AV_offset = []
        self.mean_distance = []
    #iterate through folder and load video/audio features
    #send to the quantative stats method for processing
    def processFeatures(self):
        #use for real data 
        features_folder = '/datac/nkanama/facebookDataset/output_model__real/pywork/features'
        #use for fake data
        #features_folder = '/datac/nkanama/facebookDataset/output_model_fake/pywork/features'
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
        #print(len(videoFeatures), " ", len(videoFeatures[0]))
        dists = calc_pdist(videoFeatures, audioFeatures, vshift=10)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        #finds the min distance for the video 
        minval, minidx = torch.min(mdist, 0)
        #standard offset - index of min distance 
        offset = vshift-minidx
        #confidence is median distance - min distance and represents confidence of sync error
        conf = torch.median(mdist) - minval
        #print("median ",torch.median(mdist), " min val ", minval, "confidence ", conf )
        fdist = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf = torch.median(mdist).numpy() - fdist
        fconfm = signal.medfilt(fconf, kernel_size=9)
        dists_npy = numpy.array([dist.numpy() for dist in dists])
        #median distance
        medianDistance = torch.median(mdist)
        #add for future stats 
        self.median_distance.append(medianDistance)
        self.AV_offset.append(offset)
        self.min_distance.append(minval)
        self.confidence.append(conf)
        self.mean_distance.append(torch.mean(mdist))
        
    #aggregate class member lists and find averages and print them out 
    def aggregateStats(self):
        #hard stats
        print("average min distance ", sum(self.min_distance)/len(self.min_distance))
        print("average median distance ", sum(self.median_distance)/len(self.median_distance))
        print("average mean distance ", sum(self.mean_distance)/len(self.mean_distance))
        print("average offset ", sum(self.AV_offset)//len(self.AV_offset))
        print("confidence ", sum(self.confidence)/len(self.confidence))
        
        
        #use for real data
        stats_folder = '/datac/nkanama/facebookDataset/output_model__real/pywork/stats'
        #use for fake data
        #stats_folder = '/datac/nkanama/facebookDataset/output_model_fake/pywork/stats'
        #visualization of offset over videos
        plt.plot(self.AV_offset)
        plt.xlabel("number of videos")
        plt.ylabel("audio-visual offset")
        plt.title("offset over 10 sec videos")
        plt.savefig(os.path.join(stats_folder, "VideoOffset.png"))
        plt.show()
        plt.clf()
        #visualization of median distance over videos
        plt.plot(self.median_distance)
        plt.xlabel("number of videos")
        plt.ylabel("median distance")
        plt.title("median distance for 10 sec videos")
        plt.show()
        plt.savefig(os.path.join(stats_folder, "VideoDistance.png"))
        plt.clf()
        #visualization of histogram of median distance over videos
        histogram_mean = []
        for x in range(0,21):
            histogram_mean.append(0)
        for index, element in enumerate(self.mean_distance):
            histogram_mean[int(round(float(self.mean_distance[index]),0))] = histogram_mean[int(round(float(self.mean_distance[index]),0))] + 1
        plt.plot(histogram_mean)
        plt.xlabel("mean distance values")
        plt.ylabel("frequency")
        plt.title("histogram of mean distance values")
        plt.show()
        plt.savefig(os.path.join(stats_folder, "HistogramMeanDistance.png"))
        plt.clf()
        

if __name__ == '__main__':
    #run core of stats 
    intialTest = Stats()
    intialTest.processFeatures()
    intialTest.aggregateStats()