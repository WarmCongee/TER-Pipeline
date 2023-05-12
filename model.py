from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(MLP, self).__init__()

        self.all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        self.module = nn.Sequential(*self.all_layers)
        self.fc_out_1 = nn.Linear(layers[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers[-1], output_dim2)
        
    def forward(self, inputs):
        features = self.module(inputs)
        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        return features, emos_out, vals_out



class Attention(nn.Module):
    def __init__(self, audio_dim, text_dim, video_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(Attention, self).__init__()

        self.audio_mlp = self.MLP(audio_dim, layers, dropout)
        self.text_mlp  = self.MLP(text_dim,  layers, dropout)
        self.video_mlp = self.MLP(video_dim, layers, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 3
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 3)
        self.fc_out_1 = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers_list[-1], output_dim2)
    
    def MLP(self, input_dim, layers, dropout):
        all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        module = nn.Sequential(*all_layers)
        return module

    def forward(self, audio_feat, text_feat, video_feat):
        audio_hidden = self.audio_mlp(audio_feat) # [32, 128]
        text_hidden  = self.text_mlp(text_feat)   # [32, 128]
        video_hidden = self.video_mlp(video_feat) # [32, 128]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        emos_out  = self.fc_out_1(fused_feat)
        vals_out  = self.fc_out_2(fused_feat)
        return fused_feat, emos_out, vals_out


class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1)
        target = (target.squeeze().long())
        loss = self.loss(pred, target) / len(pred)
        return loss

class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = (pred.view(-1,1))
        target = (target.view(-1,1))
        loss = self.loss(pred, target) / len(pred)
        return loss

class TextClassification(nn.Module):
    def __init__(self, checkpoint, freeze):
        super(TextClassification, self).__init__()
        self.encoder = BertModel.from_pretrained(checkpoint)
        if freeze == "1":
            self.encoder.embeddings.word_embeddings.requires_grad_(False)
        if freeze == "2":
            self.encoder.embeddings.requires_grad_(False)
        if freeze == "3":
            self.encoder.embeddings.requires_grad_(False)
            self.encoder.encoder.requires_grad_(False)
        self.attention = Attention(self.encoder.config.hidden_size, self.encoder.config.hidden_size, self.encoder.config.hidden_size, 6)
        
    def forward(self, enc_inputs, attention_mask, token_type_ids):
        outs = self.encoder(input_ids=enc_inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outs = outs.pooler_output
        outs = self.attention(outs, outs, outs)
        return outs




if __name__ == '__main__':
    pass
    # mo = TextClassification()
    # print(mo)
