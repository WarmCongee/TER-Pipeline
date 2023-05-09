# *_*coding:utf-8 *_*
import os
import sys
import socket


############ For LINUX ##############
DATA_DIR = {
	'MER2023': '/home/dataset/wyz_mer_dataset/',
    'MER2023_TER': '/home/dataset/wyz_mer_dataset/ter_dataset/',
    'REPOSITORY_PATH': '/home/wyz/MER-TER/TER-Pipeline/'
}
PATH_TO_RAW_AUDIO = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'ter_dataset/audio'),
}
PATH_TO_RAW_FACE = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
	'MER2023': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/base_text_refine_dataset_all.csv'),
}
PATH_TO_FEATURES = {
	'MER2023': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features'),
}
PATH_TO_LABEL = {
	'MER2023': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
}

PATH_TO_PRETRAINED_MODELS = '/home/wyz/MER-TER/TER-Pipeline/tools/'
PATH_TO_OPENSMILE = '/home/wyz/MER-TER/TER-Pipeline/tools/opensmile-2.3.0/'
PATH_TO_FFMPEG = '/home/wyz/MER-TER/TER-Pipeline/tools/ffmpeg-4.4.1-i686-static/ffmpeg'
PATH_TO_NOISE = '/home/wyz/MER-TER/TER-Pipeline/tools/musan/audio-select'

SAVED_ROOT = os.path.join('./saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ For Windows (openface-win) ##############
DATA_DIR_Win = {
	'MER2023': 'H:\\desktop\\Multimedia-Transformer\\MER2023-Baseline-master\\dataset-process',
}

PATH_TO_RAW_FACE_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'video'),
}

PATH_TO_FEATURES_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'features'),
}

PATH_TO_OPENFACE_Win = "H:\\desktop\\Multimedia-Transformer\\MER2023-Baseline-master\\tools\\openface_win_x64"