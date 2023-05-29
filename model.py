from transformers import BertModel
from transformers.adapters import BertAdapterModel, AdapterConfig
import transformers.adapters.composition as ac
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
        print(outs.last_hidden_state.shape)
        outs = outs.pooler_output
        
        outs = self.attention(outs, outs, outs)
        return outs




class FRA2UTT_new(nn.Module):
    def __init__(self, input_dim=1024, atsize=256, softmax_scale=0.3):
        super(FRA2UTT_new, self).__init__()
        self.atsize = atsize
        self.softmax_scale = softmax_scale
        self.attention_context_vector = nn.Parameter(torch.empty(32,atsize)) #(batch_size, feature_dim)
        nn.init.xavier_normal_(self.attention_context_vector)
        self.input_proj = nn.Linear(input_dim, self.atsize)
    
    def forward(self, input_tensor):
        input_proj = torch.tanh(self.input_proj(input_tensor))
        vector_attention = torch.bmm(input_proj, self.attention_context_vector.unsqueeze(2))
        #softmax
        vector_attention = F.softmax(self.softmax_scale*vector_attention,dim=1)
        output_vector = torch.mul(input_tensor, vector_attention)
        output_vector.squeeze()
        output_tensor = torch.sum(output_vector, dim=1, keepdim=False)
        return output_tensor
        


class AttentionNew(nn.Module):
    def __init__(self, audio_dim, text_dim, video_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(AttentionNew, self).__init__()
        self.fra2utt = FRA2UTT_new(input_dim=1024)
        self.audio_mlp = self.MLP(audio_dim, layers, dropout)
        self.text_mlp  = self.MLP(text_dim,  layers, dropout)
        self.video_mlp = self.MLP(video_dim, layers, dropout)

        layers_list = list(map(lambda x: int(x), layers.split(',')))
        hiddendim = layers_list[-1] * 3
        self.attention_mlp = self.MLP(hiddendim, layers, dropout)

        self.fc_att   = nn.Linear(layers_list[-1], 3)
        self.fc_out_e = nn.Linear(layers_list[-1], output_dim1)
        self.fc_out_v = nn.Linear(layers_list[-1], output_dim2)
        self.fc_out_ev = nn.Linear(output_dim1, output_dim2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU(num_parameters=6, init=0.25)
        self.softmax = nn.Softmax(dim=1)
        self.fc_out_att_ev = nn.Linear(2, 1)
    
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
        # audio_feat = self.softmax(audio_feat)
        # text_feat = self.softmax(text_feat)
        # video_feat = self.softmax(video_feat)
        audio_feat = self.fra2utt(audio_feat)
        video_feat = self.fra2utt(video_feat)
        text_feat = self.fra2utt(text_feat) 
        audio_hidden = self.audio_mlp(audio_feat) # [32, 128]
        text_hidden  = self.text_mlp(text_feat)   # [32, 128]
        video_hidden = self.video_mlp(video_feat) # [32, 128]
        # print(audio_hidden)
        # print(text_hidden)
        # print(video_hidden)

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1) # [32, 384]
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2) # [32, 128, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)
        fused_feat = fused_feat.squeeze() # [32, 128]
        emos_out  = self.fc_out_e(fused_feat)
        vals_out_e  = self.fc_out_ev(emos_out)
        vals_out_e = self.tanh(vals_out_e)
        vals_out_v = self.fc_out_v(fused_feat)
        vals_out_total = torch.cat([vals_out_e, vals_out_v], dim=1)
        vals_out = self.fc_out_att_ev(vals_out_total)
        return fused_feat, emos_out, vals_out

class AdapterClassification(nn.Module):
    def __init__(self, checkpoint, freeze):
        super(AdapterClassification, self).__init__()
        self.encoder = BertAdapterModel.from_pretrained(checkpoint)
        adapter_config_str = 'pfeiffer'
        if adapter_config_str not in self.encoder.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(adapter_config_str)
            # add a new adapter
            self.encoder.add_adapter(adapter_config_str, config=adapter_config)
        # Enable adapter training
        self.encoder.train_adapter(adapter_config_str)
        self.encoder.set_active_adapters(adapter_config_str)

        # if freeze == "1":
        #     self.encoder.embeddings.word_embeddings.requires_grad_(False)
        # if freeze == "2":
        #     self.encoder.embeddings.requires_grad_(False)
        # if freeze == "3":
        #     self.encoder.embeddings.requires_grad_(False)
        #     self.encoder.encoder.requires_grad_(False)
        # if freeze == "4":
        #     self.encoder.embeddings.requires_grad_(False)
        #     self.encoder.encoder.requires_grad_(False)
        #     self.encoder.pooler.requires_grad_(False)
        self.attention = AttentionNew(self.encoder.config.hidden_size, self.encoder.config.hidden_size, self.encoder.config.hidden_size, 6)
        
    def forward(self, enc_inputs, attention_mask, token_type_ids):
        outs = self.encoder(input_ids=enc_inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outs = outs.pooler_output
        outs = self.attention(outs, outs, outs)
        return outs




if __name__ == '__main__':
    pass

