#!/usr/bin/python
import sys
import time
import os
import pdb
import argparse
import pickle
import subprocess
import glob
import cv2
import ast
import numpy as np
from shutil import rmtree
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal
from detectors import S3FD
from run_pipeline import main as pipelineMain
from run_syncnet import main as syncMain
from run_visualise import main as visualMain
import json


def checkValid(name):
    json_file_1 = "/datab/nkanama/facebookDataset/dfdc_train_part_0/metadata.json"
    json_file_2 = "/datab/nkanama/facebookDataset/dfdc_train_part_9/metadata.json"
    json_file_3 = "/datab/nkanama/facebookDataset/dfdc_train_part_10/metadata.json"
    realOrFake = False
    json_check_1 = open(json_file_1)
    json_check_1X = json.load(json_check_1)
    json_check_2 = open(json_file_2)
    json_check_2X = json.load(json_check_2)
    json_check_3 = open(json_file_3)
    json_check_3X = json.load(json_check_3)
    for (key, values) in json_check_1X.items():
        if(key == name and values['label'] == 'FAKE'):
            realOrFake = True
    for (key, values) in json_check_2X.items():
        if(key == name and values['label'] == 'FAKE'):
            realOrFake = True
    for (key, values) in json_check_3X.items():
        if(key == name and values['label'] == 'FAKE'):
            realOrFake = True
    json_check_1.close()
    json_check_2.close()
    json_check_3.close()
    return realOrFake


def data_test():
    # Iterate through fake data directory and run through model
    directory = '/datab/nkanama/facebookDataset/trialRun/data'
    # output directory for fake data
    output_directory = '/datac/nkanama/facebookDataset/output_model_fake'
    # output directory for real data
    #output_directory = '/datac/nkanama/facebookDataset/output_model__real'
    videofile = ' '
    reference = ' '
    num = 0
    num_videos = 0
    # iterating over videos
    for video in os.listdir(directory):
        # iterates through every 10th video
        # skips directories and fake video
        if (os.path.isdir(video) or checkValid(video)):
            continue
        num = num + 1
        if(num < 5):
            continue
        else:
            num_videos = num_videos+1
            num = 0
        if(num_videos == 15):
            break

        videofile = os.path.join(directory, video)
        cropped_string = video.find('.')
        reference = video[0:cropped_string]
        # create argument parses
        opt_pipeline = get_args_pipeline(
            output_directory, videofile, reference)
        # run through pipeline script to process videos
        pipelineMain(opt_pipeline)
        # run through syncnet script to process videos
        opt_syncnet = get_args_syncnet(output_directory, videofile, reference)
        syncMain(opt_syncnet)
        # run through visualize script to process videos
        opt_visual = get_visual_args(output_directory, videofile, reference)
        visualMain(opt_visual)


def get_args_pipeline(data_dir, videofile, reference):
    parser = argparse.ArgumentParser(description="FaceTracker")
    parser.add_argument('--data_dir',       type=str,
                        default=data_dir, help='Output directory')
    parser.add_argument('--videofile',      type=str,
                        default=videofile,   help='Input video file')
    parser.add_argument('--reference',      type=str,
                        default=reference,   help='Video reference')
    parser.add_argument('--facedet_scale',  type=float,
                        default=0.25, help='Scale factor for face detection')
    parser.add_argument('--crop_scale',     type=float,
                        default=0.40, help='Scale bounding box')
    parser.add_argument('--min_track',      type=int,
                        default=100,  help='Minimum facetrack duration')
    parser.add_argument('--frame_rate',     type=int,
                        default=25,   help='Frame rate')
    parser.add_argument('--num_failed_det', type=int, default=25,
                        help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--min_face_size',  type=int,
                        default=100,  help='Minimum face size in pixels')
    opt = parser.parse_args()

    setattr(opt, 'avi_dir', os.path.join(opt.data_dir, 'pyavi'))
    setattr(opt, 'tmp_dir', os.path.join(opt.data_dir, 'pytmp'))
    setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))
    setattr(opt, 'crop_dir', os.path.join(opt.data_dir, 'pycrop'))
    setattr(opt, 'frames_dir', os.path.join(opt.data_dir, 'pyframes'))
    return opt


def get_args_syncnet(data_dir, videofile, reference):
    # ==================== PARSE ARGUMENT ====================

    parser = argparse.ArgumentParser(description="SyncNet")
    parser.add_argument('--initial_model', type=str,
                        default="data/syncnet_v2.model", help='')
    parser.add_argument('--batch_size', type=int, default='20', help='')
    parser.add_argument('--vshift', type=int, default='15', help='')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='')
    parser.add_argument('--videofile', type=str, default=videofile, help='')
    parser.add_argument('--reference', type=str, default=reference, help='')
    opt = parser.parse_args()

    setattr(opt, 'avi_dir', os.path.join(opt.data_dir, 'pyavi'))
    setattr(opt, 'tmp_dir', os.path.join(opt.data_dir, 'pytmp'))
    setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))
    setattr(opt, 'crop_dir', os.path.join(opt.data_dir, 'pycrop'))
    return opt


def get_visual_args(data_dir, videofile, reference):
	# ==================== PARSE ARGUMENT ====================

	parser = argparse.ArgumentParser(description="SyncNet")
	parser.add_argument('--data_dir', 	type=str, default='data/work', help='')
	parser.add_argument('--videofile', 	type=str, default='', help='')
	parser.add_argument('--reference', 	type=str, default='', help='')
	parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate')
	opt = parser.parse_args()

	setattr(opt, 'avi_dir', os.path.join(opt.data_dir, 'pyavi'))
	setattr(opt, 'tmp_dir', os.path.join(opt.data_dir, 'pytmp'))
	setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))
	setattr(opt, 'crop_dir', os.path.join(opt.data_dir, 'pycrop'))
	setattr(opt, 'frames_dir', os.path.join(opt.data_dir, 'pyframes'))
	return opt
if __name__ == '__main__':
    data_test()
