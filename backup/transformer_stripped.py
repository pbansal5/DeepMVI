import torch
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import _pickle as cPickle
import random
import math,copy
import torch.nn.functional as F
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

    
class TransformerConvModel(nn.Module):
    def __init__(self,ninp=64, nhead=2, nhid=64, nlayers=4,dropout=0.5):
        super(TransformerConvModel, self).__init__()
        from layer_transformer import TransformerEncoderLayer,TransformerEncoder
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder_forward = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder_backward = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder_mean = nn.Linear(ninp, 1)
        self.decoder_std = nn.Linear(ninp, 1)
        self.mse_loss = nn.MSELoss()
        self.mask_len = 10
        
    def forward(self, src_forward_in,src_backward_in,query_forward,query_backward,mask):
        src_forward = self.pos_encoder(src_forward_in.transpose(1,2)).transpose(0,1)
        src_backward = self.pos_encoder(src_backward_in.transpose(1,2)).transpose(0,1)
        query_forward = self.pos_encoder(query_forward.transpose(1,2)).transpose(0,1)
        query_backward = self.pos_encoder(query_backward.transpose(1,2)).transpose(0,1)
        mask_forward,mask_backward,mask = self._generate_square_subsequent_mask(src_forward,mask,self.training)
        output_forward = self.transformer_encoder_forward(src_forward,query_forward,mask_forward).clamp(min=0)
        output_backward = self.transformer_encoder_backward(src_backward,query_backward,mask_backward).clamp(min=0)
        output = torch.cat([output_forward,output_backward],dim=2)
        return output.transpose(0,1),mask

    def _generate_square_subsequent_mask(self,sz,loss_mask,train: bool):
        sz = len(sz)
        mask_len = self.mask_len
        backward_mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        backward_mask = ~backward_mask
        forward_mask = (torch.triu(torch.ones(sz, sz)) != 1)

        if (train):
            random_points = torch.randint(high=mask_len,size=(backward_mask.shape[0],))
            #random_points = torch.zeros(backward_mask.shape[0])+10
            start_points = (torch.arange(backward_mask.shape[0])-random_points).clamp(min=0).clamp(max=backward_mask.shape[0]-mask_len)
            for x in range(mask_len):
                backward_mask[torch.arange(backward_mask.shape[0]).long(),start_points.long()+x] = False
                forward_mask[torch.arange(forward_mask.shape[0]).long(),start_points.long()+x] = False

        to_remove = torch.nonzero(torch.logical_or(~backward_mask.any(dim=1),~forward_mask.any(dim=1)).float())
        loss_mask[:,to_remove] = 0

        backward_mask = backward_mask.float().masked_fill(backward_mask == 0, float('-inf')).masked_fill(backward_mask == 1, float(0.0))
        forward_mask = forward_mask.float().masked_fill(forward_mask == 0, float('-inf')).masked_fill(forward_mask == 1, float(0.0))
        backward_mask[to_remove,to_remove] = 0
        forward_mask[to_remove,to_remove] = 0
        return forward_mask.to(loss_mask.device),backward_mask.to(loss_mask.device),loss_mask

    
class OurModel(nn.Module):
    def __init__(self,sizes,embedding_size=16,time_embed=32,time_len=1000,nkernel=64,nlayers=4,nhid=32,nhead=2,kernel_size=5,residuals=None):
        super(OurModel, self).__init__()
        hidden_dim = 256
        self.k = 20
        self.tau = 1
        self.embeddings = []
        for x in sizes:
            self.embeddings.append(nn.Embedding(x, embedding_size))
        self.transformer_embeddings = nn.ModuleList(self.embeddings)
        self.time_embeddings = nn.Embedding(time_len, time_embed)
        num_feats = embedding_size*len(sizes)+nkernel+time_embed
        
        if (nkernel != 0):
            self.conv_forward = nn.Conv1d(1,nkernel,kernel_size = kernel_size)
            self.conv_backward = nn.Conv1d(1,nkernel,kernel_size = kernel_size)
            self.pad_forward = torch.nn.ConstantPad1d((kernel_size-1,0),0.0)
            self.pad_backward = torch.nn.ConstantPad1d((0,kernel_size-1),0.0)
            self.transformer = TransformerConvModel(nlayers=nlayers,nhid=nhid,ninp=num_feats,nhead=nhead,dropout=0.1)
            self.outlier_layer1 = nn.Linear(3*len(sizes)+2*num_feats,hidden_dim)
        else :
            self.outlier_layer1 = nn.Linear(3*len(sizes)+num_feats,hidden_dim)
            
        self.outlier_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.mean_outlier_layer = nn.Linear(hidden_dim,1)
        self.residuals = residuals
        self.dropout = torch.nn.Dropout(p=0.2)
        
    def compute_feats(self,y_context : List[torch.Tensor]):
        final1,final2,final3 = [],[],[]
        for i,embed in enumerate(self.transformer_embeddings):
            temp = torch.cdist(embed.weight.unsqueeze(0),embed.weight.unsqueeze(0),p=2.0)[0]+1e-3
            temp[torch.arange(temp.shape[0]),torch.arange(temp.shape[1])] = 0
            temp_w = torch.exp(-temp/self.tau)
            temp_i = torch.argsort(temp_w)[:,-(self.k+1):-1]
            #temp_i = torch.argsort(temp_w)[:,:-1]
            temp_s = y_context[i][torch.arange(y_context[i].shape[0])[:,None],:,temp_i]
            temp_w = temp_w[torch.arange(temp_w.shape[0])[:,None],None,temp_i].repeat(1,1,y_context[0].shape[1])
            temp_w[temp_s==0] = temp_w[temp_s==0]/1e3
            final1.append(temp_w.sum(dim=1,keepdim=True))
            final2.append((temp_w*temp_s).sum(dim=1,keepdim=True)/temp_w.sum(dim=1,keepdim=True))
            final3.append(torch.std(temp_s,dim=1,keepdim=True))
        return torch.cat(final1+final2+final3,dim=1).transpose(1,2)

    def core(self,in_series,mask,residuals):
        embeddings = []
        for i,embed in enumerate(self.transformer_embeddings):
            temp = embed.weight#[context_info[0][:,i]]
            embeddings.append(temp.unsqueeze(1).repeat(1,in_series.shape[1],1))
            
        temp = self.time_embeddings.weight.unsqueeze(0).repeat(in_series.shape[0],1,1).to(in_series.device)
        embeddings.append(temp)
        hidden_state = torch.cat(embeddings,dim=2)
        temp = self.compute_feats([residuals])
        feats = self.dropout(torch.cat([temp,hidden_state],dim=2))
        feats = self.outlier_layer1(feats).clamp(min=0)
        feats = self.outlier_layer2(feats).clamp(min=0)
        mean = self.mean_outlier_layer(feats)[:,:,0]
        return mean,mask

    def forward (self,in_series,mask,residuals):
        mean,mask = self.core(in_series,mask,residuals)
        return {'mae':self.mae_loss(mean,in_series,mask).mean()}
    
    @torch.jit.export
    def validate(self,in_series,mask,residuals):
        mean,mask = self.core(in_series,mask,residuals)
        loss = self.mae_loss(mean,in_series,mask)
        return {'loss_values':loss,'values':mean}


    def mae_loss(self,y,y_pred,mask):
        temp = torch.abs((y_pred-y).cuda())
        loss = temp[mask>0]
        return loss

def make_validation (matrix,block_size):
    validation_points = np.random.uniform(0,matrix.shape[0]-block_size,(matrix.shape[1])).astype(np.int)
    train_matrix = copy.deepcopy(matrix)
    for i,x in enumerate(validation_points) :
        train_matrix[x:x+block_size,i] = np.nan
    return train_matrix.transpose(),matrix.transpose()

def train(model,train_feats,val_feats,device):
    best_state_dict = model.state_dict()
    best_loss = float('inf')

    lr = 1e-2
    optim = torch.optim.Adam(model.parameters(),lr=lr)

    tolerance_epoch = 0
    patience  = 5
    max_epoch = 1000
    train_mask = np.ones(train_feats.shape)
    train_mask[np.isnan(train_feats)] = 0
    val_mask = np.zeros(val_feats.shape)
    val_mask[np.isnan(train_feats)] = 1
    val_mask[np.isnan(val_feats)] = 0
    train_mask = torch.from_numpy(train_mask).float().to(device)
    val_mask = torch.from_numpy(val_mask).float().to(device)
    train_inp_ = torch.from_numpy(np.nan_to_num(train_feats,0)).float().to(device)
    val_inp_ = torch.from_numpy(np.nan_to_num(val_feats,0)).float().to(device)
    residuals = train_inp_.detach().clone().unsqueeze(2).repeat(1,1,train_inp_.shape[0]).float().to(device)
    for epoch in range(max_epoch):
        print ("Starting Epoch : %d"%epoch)
        train_inp = train_inp_ + torch.randn(train_inp_.shape[0]).to(device)[:,None]/10
        train_mask[torch.bernoulli(torch.zeros(train_mask.shape)+0.1).to(device).bool()] = 0
        loss = model(train_inp,train_mask,residuals)
        optim.zero_grad()
        print (loss['mae'])
        loss['mae'].backward()
        optim.step()
        if (epoch % 5 == 0):
            model.eval()
            with torch.no_grad():
                loss = model.validate(val_inp_,val_mask,residuals)
                loss_mre_num = loss['loss_values'].sum()
                count = len(loss['loss_values'])
            if (float(loss_mre_num)/count < 0.95*best_loss):
                tolerance_epoch = 0
                best_loss = float(loss_mre_num)/count
                best_state_dict = model.state_dict()
            else :
                tolerance_epoch += 1
            print ('done validation, Patience : ',tolerance_epoch)
            print ('validation loss : ',float(loss_mre_num/count))
            model.train()
            if (tolerance_epoch == patience):
                print ('Early Stopping')
                return best_state_dict
    return best_state_dict

def test(model,val_feats,device):
    output_matrix = copy.deepcopy(val_feats)
    model.eval()
    with torch.no_grad():
        mask = torch.zeros(val_feats.shape)
        mask[np.isnan(val_feats)] = 1
        inp_ = torch.from_numpy(np.nan_to_num(val_feats,0)).float()
        residuals = inp_.detach().clone().unsqueeze(2).repeat(1,1,inp_.shape[0])
        loss = model.validate(inp_.to(device),mask.to(device),residuals.to(device))
        output_matrix[:,:] = np.where(mask.detach().cpu().numpy()[0],loss['values'].detach().cpu().numpy()[0],output_matrix[:,:])
    model.train()
    return output_matrix


def transformer_recovery(input_feats):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device('cuda:%d'%0)

    block_size = 10
    train_feats,val_feats = make_validation(input_feats,block_size)

    # drift10,airq
    model = OurModel(sizes=[train_feats.shape[0]],nkernel=0,embedding_size=8,time_embed=16,nhid=16,nlayers=1,nhead=2,time_len=train_feats.shape[1]).to(device)
    # model = OurModel(sizes=[train_feats.shape[1]],nkernel=8,embedding_size=8,time_embed=16,nhid=16,nlayers=1,nhead=2,time_len=train_feats.shape[0]).to(device)
    # chlorine
    # model = OurModel(sizes=[train_feats.shape[1]],nkernel=8,embedding_size=8,time_embed=8,nhid=16,nlayers=1,nhead=2,time_len=train_feats.shape[0]).to(device)
    # model.transformer.mask_len = block_size
    # model = torch.jit.script(model)

    best_state_dict = train(model,train_feats,val_feats,device)
    model.load_state_dict(best_state_dict)

    matrix = test(model,val_feats,device)
#    print (np.where(np.isnan(matrix)))
    return matrix
