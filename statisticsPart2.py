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
    def __init__(self, mean_distance=None, numCorrect=0):
        self.numCorrect = 0
        self.mean_distance = []
    #iterate through folder and load video/audio features
    #send to the quantative stats method for processing

    def processFeatures(self):
        #use for real data
        features_folder_real = '/datac/nkanama/facebookDataset/output_model__real/pywork/features'
        #use for fake data
        features_folder_fake = '/datac/nkanama/facebookDataset/output_model_fake/pywork/features'

        for directory in os.listdir(features_folder_real):
            #cpu
            audio_features = torch.load(os.path.join(
                features_folder_real, directory, 'audioFeatures.pt'))
            video_features = torch.load(os.path.join(
                features_folder_real, directory, 'videoFeatures.pt'))
            self.quantStats(audio_features, video_features, True)
        for directory in os.listdir(features_folder_fake):
            #cpu
            audio_features = torch.load(os.path.join(
                features_folder_fake, directory, 'audioFeatures.pt'))
            video_features = torch.load(os.path.join(
                features_folder_fake, directory, 'videoFeatures.pt'))
            self.quantStats(audio_features, video_features, False)

    #calculate median, min distance and confidence and AV offset for audio/video features
    #add to storage in class member variables
    def quantStats(self, audioFeatures, videoFeatures, realOrFake):
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

        #mean distance
        self.mean_distance.append(torch.mean(mdist))
        if(realOrFake == True and self.mean_distance[-1] < 11.7):
            self.numCorrect = self.numCorrect + 1
        elif(realOrFake == False and self.mean_distance[-1] >= 11.7):
            self.numCorrect = self.numCorrect + 1

    #aggregate class member lists and find averages and print them out

    def aggregateStats(self):
        print("percent correct detections ",
              self.numCorrect/len(self.mean_distance))


if __name__ == '__main__':
    #run core of stats
    intialTest = Stats()
    intialTest.processFeatures()
    intialTest.aggregateStats()
