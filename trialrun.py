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
from run_pipeline import main


def data_test():
    # Iterate through fake data directory and run through model
    directory = '/datab/nkanama/facebookDataset/trailRun/data'
    output_directory = '/datab/nkanama/facebookDataset/trailRun/model_output'
    videofile = ' '
    reference = ' '
    num = 0
    # iterating over videos
    for video in directory:
        # iterates through every 10th viddeo
        num = num + 1
        if(num < 10):
            continue
        else:
            num = 0
        videofile = video
        cropped_string = video.find('.')
        reference = video[0:cropped_string]
        # create argument parses
        opt = get_args(output_directory, videofile, reference)
        # run through pipeline script to process videos
        main(opt)
        break


def get_args(data_dir, videofile, reference):
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

    setattr(opt, 'avi_dir', os.path.join(
        opt.data_dir, opt.videofile + 'pyavi'))
    setattr(opt, 'tmp_dir', os.path.join(
        opt.data_dir, opt.videofile + 'pytmp'))
    setattr(opt, 'work_dir', os.path.join(
        opt.data_dir, opt.videofile + 'pywork'))
    setattr(opt, 'crop_dir', os.path.join(
        opt.data_dir, opt.videofile + 'pycrop'))
    setattr(opt, 'frames_dir', os.path.join(
        opt.data_dir, opt.videofile + 'pyframes'))
    return opt


if __name__ == '__main__':
    data_test()
