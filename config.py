# *_*coding:utf-8 *_*
import os
import sys
import socket


############ For LINUX ##############
DATA_DIR = {
	'MER2023': '/home/dataset/wyz_mer_dataset/',
    'REPOSITORY_PATH': '/home/wyz/MER-TER/TER-Pipeline/',
    'MER2023_SIMPLE_TRANS': '/home/wyz/MER-TER/TER-Pipeline/dataset/',
    'MER2023_LONG_TRANS': '/home/wyz/MER-TER/TER-Pipeline/',
    'MER2023_WHISPER_LARGE2_TRANS': '/home/wyz/MER-TER/TER-Pipeline/',
    'BASE_MOBILE': '/home/wyz/MER-TER/TER-Pipeline/',
    'NEW_MOBILE': '/home/wyz/MER-TER/TER-Pipeline/'
}
PATH_TO_RAW_AUDIO = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'ter_dataset/audio'),
}
PATH_TO_RAW_FACE = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
	'MER2023': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/base_text_refine_dataset_all.csv'),
    'MER2023_SIMPLE_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/base_text_dataset_all.csv'),
    'MER2023_LONG_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/base_long_text.csv'),
    'MER2023_WHISPER_LARGE2_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/whisper_text_dataset_all.csv'),
    'BASE_MOBILE': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/base_mobile_text.csv'),
    'NEW_MOBILE': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/new_mobile_text_refine.csv'),
    'BASE_TRAIN': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/base_train.csv'),
    'BASE_TEST': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/base_test.csv'),
}
PATH_TO_FEATURES = {
	'MER2023': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features'),
    'MER2023_SIMPLE_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/simple'),
    'MER2023_LONG_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/long'),
    'MER2023_WHISPER_LARGE2_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper'),
    'BASE_MOBILE': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/mobile'),
    'NEW_MOBILE': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/new_mobile'),
    
    'MER2023_WHISPER_LARGE2_TRANS_-6_-3': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_-6_-3'),
    'MER2023_WHISPER_LARGE2_TRANS_-8_-5': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_-8_-5'),
    'MER2023_WHISPER_LARGE2_TRANS_-2_-1': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_-2_-1'),
    'MER2023_WHISPER_LARGE2_TRANS_-4_-3': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_-4_-3'),
    'MER2023_WHISPER_LARGE2_TRANS_-8_-7': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_-8_-7'),

    'MER2023_WHISPER_LARGE2_TRANS_1_4': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_1_4'),
    'MER2023_WHISPER_LARGE2_TRANS_5_8': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_5_8'),
    'MER2023_WHISPER_LARGE2_TRANS_9_12': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_9_12'),
    'MER2023_WHISPER_LARGE2_TRANS_13_16': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'features/whisper_old_13_16'),
}
PATH_TO_LABEL = {
	'MER2023': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_SIMPLE_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_LONG_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label_base_long_text.npz'),
    'MER2023_WHISPER_LARGE2_TRANS': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'BASE_MOBILE': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/base_mobile_text.npz'),
    'NEW_MOBILE': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/new_mobile_text.npz'),

    'MER2023_WHISPER_LARGE2_TRANS_-6_-3': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_WHISPER_LARGE2_TRANS_-8_-5': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_WHISPER_LARGE2_TRANS_-2_-1': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_WHISPER_LARGE2_TRANS_-4_-3': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_WHISPER_LARGE2_TRANS_-8_-7': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),

    'MER2023_WHISPER_LARGE2_TRANS_1_4': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_WHISPER_LARGE2_TRANS_5_8': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_WHISPER_LARGE2_TRANS_9_12': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),
    'MER2023_WHISPER_LARGE2_TRANS_13_16': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'dataset/label-6way.npz'),

}

PATH_TO_RESULT = {
	'RESULT_CSV': os.path.join(DATA_DIR['REPOSITORY_PATH'], 'result/results.txt'),
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