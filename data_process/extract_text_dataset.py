import re
import os
import sys
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd

import cv2 # pip install opencv-python
import config_path

# split audios from videos
def split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for video_path in tqdm.tqdm(glob.glob(video_root+'/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(save_root, videoname + '.wav')
        if os.path.exists(audio_path): continue
        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config_path.PATH_TO_FFMPEG, video_path, audio_path)
        os.system(cmd)

# preprocess dataset-release
def normalize_dataset_format(data_root, save_root):
    ## input path
    train_data  = os.path.join(data_root, 'train')
    train_label = os.path.join(data_root, 'train-label.csv')

    ## output path
    save_video = os.path.join(save_root, 'video')
    save_label = os.path.join(save_root, 'label-6way.npz')
    if not os.path.exists(save_root): os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## move all videos into save_video
    for temp_root in [train_data]:
        video_paths = glob.glob(temp_root + '/*')
        for video_path in tqdm.tqdm(video_paths):
            video_name = os.path.basename(video_path)
            new_path = os.path.join(save_video, video_name)
            shutil.copy(video_path, new_path)

    ## generate label path
    train_corpus = {}
    df_label = pd.read_csv(train_label)
    for _, row in df_label.iterrows(): ## read for each row
        name = row['name']
        emo  = row['discrete']
        val  = row['valence']
        train_corpus[name] = {'emo': emo, 'val': val}

    np.savez_compressed(save_label,
                        train_corpus=train_corpus)

# generate transcription files using asr
def generate_transcription_files_asr(audio_root: str, save_path: str):
    import torch
    import wenetruntime as wenet
    decoder = wenet.Decoder("/home/wyz/MER-TER/TER-Pipeline/tools/wenet/wenetspeech_u2pp_conformer_libtorch", lang='chs')
    
    names = []
    sentences = []
    emos =[]
    valences =[]

    lable_dictionary = np.load(config_path.TRAIN_LABLE_NPZ_PATH, allow_pickle=True)

    for audio_path in tqdm.tqdm(glob.glob(audio_root + '/*')):
        name = os.path.basename(audio_path)[:-4]
        print(audio_path)
        sentence = decoder.decode_wav(audio_path)
        sentence = sentence.split('"')[5]
        names.append(name)
        sentences.append(sentence)
        
        emos.append(lable_dictionary['train_corpus'][()][name]['emo'])
        valences.append(lable_dictionary['train_corpus'][()][name]['val'])

    ## write to csv file
    columns = ['emo', 'val', 'name', 'sentence']
    data = np.column_stack([emos, valences, names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(save_path, sep='\t', index=False)

# add punctuation to transcripts
def refinement_transcription_files_asr(old_path, new_path):
    from paddlespeech.cli.text.infer import TextExecutor
    text_punc = TextExecutor()

    ## read 
    emos, vals, names, sentences = [], [], [], []
    df_label = pd.read_csv(old_path, sep = '\t')
    for _, row in df_label.iterrows():
        emos.append(row['emo'])
        vals.append(row['val'])
        names.append(row['name'])
        sentence = row['sentence']
        if pd.isna(sentence):
            sentences.append('')
        else:
            sentence = text_punc(text=sentence)
            sentences.append(sentence)
        print (sentences[-1])

    ## write
    columns = ['emo', 'val', 'name', 'sentence']
    data = np.column_stack([emos, vals, names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(new_path, sep='\t', index=False)

def divide_train_test_dateset(csv_path_all, label_path_all, \
                               csv_save_path_train, label_save_path_train, \
                              csv_save_path_test, label_save_path_test):
    emos, vals, names, sentences = [], [], [], []
    df_label = pd.read_csv(csv_path_all, sep = '\t')
    
    # TODO: Train里面的数据集分割出Test


def preserve_longer_text_datasets(csv_path_all, save_path, label_root_path):
    emos, vals, names, sentences = [], [], [], []
    train_corpus = {}
    df_label = pd.read_csv(csv_path_all, sep = '\t')
    for _, row in df_label.iterrows():
        sentence = row['sentence']
        if (not pd.isna(sentence)) and len(sentence) > 5:
            emos.append(row['emo'])
            vals.append(row['val'])
            names.append(row['name'])
            train_corpus[row['name']] = {'emo': row['emo'], 'val': row['val']}
            sentences.append(sentence)
    columns = ['emo', 'val', 'name', 'sentence']
    data = np.column_stack([emos, vals, names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(save_path, sep='\t', index=False)

    ## output path
    save_label = os.path.join(label_root_path, 'label_base_long_text.npz')
    if not os.path.exists(label_root_path): os.makedirs(label_root_path)
    np.savez_compressed(save_label,
                        train_corpus=train_corpus)
    

# generate transcription files using asr
def generate_transcription_files_whisper(audio_root: str, save_path: str):
    import whisper
    whisper_model = whisper.load_model("large-v2", device="cpu")
    
    whisper_model.encoder.to("cuda:0")
    whisper_model.decoder.to("cuda:1")
    whisper_model.decoder.register_forward_pre_hook(lambda _, inputs: tuple([inputs[0].to("cuda:1"), inputs[1].to("cuda:1")] + list(inputs[2:])))
    whisper_model.decoder.register_forward_hook(lambda _, inputs, outputs: outputs.to("cuda:0"))

    names = []
    sentences = []
    emos =[]
    valences =[]

    lable_dictionary = np.load(config_path.TRAIN_LABLE_NPZ_PATH, allow_pickle=True)

    for audio_path in tqdm.tqdm(glob.glob(audio_root + '/*')):
        name = os.path.basename(audio_path)[:-4]
        print(audio_path)
        sentence = (whisper_model.transcribe(audio_path, language='zh', initial_prompt='你好，以下是普通话的句子。'))['text']
        # sentence = sentence.split('"')[5]
        names.append(name)
        sentences.append(sentence)
        
        emos.append(lable_dictionary['train_corpus'][()][name]['emo'])
        valences.append(lable_dictionary['train_corpus'][()][name]['val'])

    ## write to csv file
    columns = ['emo', 'val', 'name', 'sentence']
    data = np.column_stack([emos, valences, names, sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(save_path, sep='\t', index=False)
        

def construct_text_dataset():
    # # 整理video，整理Label存储格式
    # normalize_dataset_format(config_path.MER_DATASET_PATH_SOURCE, config_path.TER_DATASET_PATH)
    # # 提取wav
    # split_audio_from_video_16k(config_path.TER_DATASET_PATH_VIDEO, config_path.TER_DATASET_PATH_AUDIO)
    # # 翻译wav得到Text，合并Text和Label到同一csv
    # generate_transcription_files_asr(config_path.TER_DATASET_PATH_AUDIO, config_path.ASR_TRANS_DATASET_PATH)
    # # 微调csv中Text结果，补充标点符号
    # refinement_transcription_files_asr(config_path.ASR_TRANS_DATASET_PATH, config_path.ASR_TRANS_REFINEMENT_DATASET_PATH)
    # # 初步划分Train和Test的数据集
    # divide_train_test_dateset(config_path.ASR_TRANS_REFINEMENT_DATASET_PATH, config_path.TRAIN_LABLE_NPZ_PATH, 
    #                           config_path.ASR_TRANS_REFINEMENT_DATASET_PATH_TRAIN, config_path.TRAIN_LABLE_NPZ_PATH_TRAIN,
    #                           config_path.ASR_TRANS_REFINEMENT_DATASET_PATH_TEST, config_path.TRAIN_LABLE_NPZ_PATH_TEST)
    # preserve_longer_text_datasets(config_path.ASR_TRANS_REFINEMENT_DATASET_PATH, config_path.ASR_TRANS_LONG_TEXT_PATH, config_path.TRAIN_LONG_LABLE_NPZ_PATH)

    # 使用whisper提取
    generate_transcription_files_whisper(config_path.TER_DATASET_PATH_AUDIO, config_path.ASR_WHISPER_TRANS_DATASET_PATH)

if __name__ == '__main__':
    construct_text_dataset()