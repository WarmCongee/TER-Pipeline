
import os
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..") 
import config

def write_log(*t, path = "./log.txt"): 
    t = ",".join([str(item) for item in t])
    f = open(path, "a")
    f.write(t + '\n')
    f.close()


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1)
        target = (target.squeeze().long())
        loss = self.loss(pred, target) / len(pred)
        return loss


def analyzing_asr_impact(pre_results, label_path, trans_path, emo2idx, idx2emo, args):
    folder_one_result = pre_results[0]
    emo_probs = (folder_one_result[f'{args.test_sets[0]}_emoprobs']).tolist()
    val_preds = folder_one_result[f'{args.test_sets[0]}_valpreds']
    names = folder_one_result[f'{args.test_sets[0]}_names']

    emo_labels = folder_one_result[f'{args.test_sets[0]}_emolabels']
    val_labels = folder_one_result[f'{args.test_sets[0]}_vallabels']

    good_name_list = {}
    mid_name_list = {}
    bad_name_list = {}

    
    for ii in range(len(names)):
        emos_pred_str = idx2emo[emo_probs[ii].index(max(emo_probs[ii]))]
        emos_label_str = idx2emo[emo_labels[ii]]
        if emo_probs[ii].index(max(emo_probs[ii]))== emo_labels[ii] and abs(val_labels[ii]-val_preds[ii]) <= 1:
            good_name_list[names[ii]] = [emos_pred_str, emos_label_str, val_preds[ii], val_labels[ii]]
        elif emo_probs[ii].index(max(emo_probs[ii])) != emo_labels[ii] and abs(val_labels[ii]-val_preds[ii]) >= 2:
            bad_name_list[names[ii]] = [emos_pred_str, emos_label_str, val_preds[ii], val_labels[ii]]
        else:
            mid_name_list[names[ii]] = [emos_pred_str, emos_label_str, val_preds[ii], val_labels[ii]]
    print(len(good_name_list))
    print(len(mid_name_list))
    print(len(bad_name_list))

    length = 50

    good_name_key_list = good_name_list.keys()
    bad_name_key_list = bad_name_list.keys()

    random_good = random.sample(good_name_key_list, length)
    random_bad = random.sample(bad_name_key_list, length)

    g_names, g_emos_p, g_emos, g_vals_p, g_vals, g_sentences  = [], [], [], [], [], []
    b_names, b_emos_p, b_emos, b_vals_p, b_vals, b_sentences  = [], [], [], [], [], []

    df_label = pd.read_csv(trans_path, sep = '\t')
    for _, row in df_label.iterrows():
        name_temp = row['name']
        if name_temp in random_good:
            g_names.append(name_temp)
            g_emos_p.append(good_name_list[name_temp][0])
            g_emos.append(good_name_list[name_temp][1])
            g_vals_p.append(good_name_list[name_temp][2])
            g_vals.append(good_name_list[name_temp][3])
            g_sentences.append(row['sentence'])
        

    columns = ['name', 'emp_pred', 'emo', 'val_pred', 'val', 'sentence']
    data = np.column_stack([g_names, g_emos_p, g_emos, g_vals_p, g_vals, g_sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv('./good.csv', sep='\t', index=False)


    df_label = pd.read_csv(trans_path, sep = '\t')
    for _, row in df_label.iterrows():
        name_temp = row['name']
        if name_temp in random_bad:
            b_names.append(name_temp)
            b_emos_p.append(bad_name_list[name_temp][0])
            b_emos.append(bad_name_list[name_temp][1])
            b_vals_p.append(bad_name_list[name_temp][2])
            b_vals.append(bad_name_list[name_temp][3])
            b_sentences.append(row['sentence'])
        

    columns = ['name', 'emp_pred', 'emo', 'val_pred', 'val', 'sentence']
    data = np.column_stack([b_names, b_emos_p, b_emos, b_vals_p, b_vals, b_sentences])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv('./bad.csv', sep='\t', index=False)

    # ## output path
    # save_label = os.path.join(label_root_path, 'label_base_long_text.npz')
    # if not os.path.exists(label_root_path): os.makedirs(label_root_path)
    # np.savez_compressed(save_label,
    #                     train_corpus=train_corpus)


# 将tencent返回结果重新组织格式
# def refact_tencent_transcription_files(source_file: str, save_path: str):
    
#     names = []
#     sentences = []
#     emos =[]
#     valences =[]

#     lable_dictionary = np.load(config.PATH_TO_LABEL['MER2023'], allow_pickle=True)
    
#     train_corpus = {}
#     name_sentence_dec = {}
#     f = open(source_file,"r") 
#     lines = f.readlines()      #读取全部内容 ，并以列表方式返回
#     for line in lines: 
#         if(line != lines[-1]):
#             temp = line.rsplit(" ",1)
#             if len(temp) == 1:
#                 sentence = ""
#                 name = temp[0].replace("(","").replace(")","").replace("\n","")
#             else:
#                 sentence = temp[0]
#                 name = temp[1].replace("(","").replace(")","").replace("\n","")
#                 sentence = sentence.replace(" ","")
#             name_sentence_dec[name] = sentence

#     for item in lable_dictionary['train_corpus'][()].keys():
#         if(item in name_sentence_dec.keys()):
#             names.append(item)
#             sentences.append(name_sentence_dec[item])
#             emos.append(lable_dictionary['train_corpus'][()][item]['emo'])
#             valences.append(lable_dictionary['train_corpus'][()][item]['val'])
#             train_corpus[item] = {'emo': lable_dictionary['train_corpus'][()][item]['emo'], 'val': lable_dictionary['train_corpus'][()][item]['val']}
#         else: 
#             names.append(item)
#             sentences.append("")
#             emos.append(lable_dictionary['train_corpus'][()][item]['emo'])
#             valences.append(lable_dictionary['train_corpus'][()][item]['val'])
#             train_corpus[item] = {'emo': lable_dictionary['train_corpus'][()][item]['emo'], 'val': lable_dictionary['train_corpus'][()][item]['val']}

#     ## write to csv file
#     columns = ['emo', 'val', 'name', 'sentence']
#     data = np.column_stack([emos, valences, names, sentences])
#     df = pd.DataFrame(data=data, columns=columns)
#     df[columns] = df[columns].astype(str)
#     df.to_csv(save_path, sep='\t', index=False)

#     save_label = config.PATH_TO_LABEL['BASE_MOBILE']
#     # if not os.path.exists('/home/wyz/MER-TER/TER-Pipeline/dataset/'): os.makedirs('/home/wyz/MER-TER/TER-Pipeline/dataset/')
#     np.savez_compressed(save_label,
#                         train_corpus=train_corpus)


if __name__ == '__main__':
    pass
    # refact_tencent_transcription_files('../dataset/rec-zkd.txt', config.PATH_TO_TRANSCRIPTIONS['BASE_MOBILE'])
    
    