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
        self.conv_forward = nn.Conv1d(1,nkernel,kernel_size = kernel_size)
        self.conv_backward = nn.Conv1d(1,nkernel,kernel_size = kernel_size)
        self.time_embeddings = nn.Embedding(time_len, time_embed)
        self.pad_forward = torch.nn.ConstantPad1d((kernel_size-1,0),0.0)
        self.pad_backward = torch.nn.ConstantPad1d((0,kernel_size-1),0.0)
        num_feats = embedding_size*len(sizes)+nkernel+time_embed
        self.transformer = TransformerConvModel(nlayers=nlayers,nhid=nhid,ninp=num_feats,nhead=nhead,dropout=0.1)
        self.outlier_layer1 = nn.Linear(3*len(sizes)+2*num_feats,hidden_dim)
        self.outlier_layer2 = nn.Linear(hidden_dim,hidden_dim)
        self.mean_outlier_layer = nn.Linear(hidden_dim,1)
        self.residuals = residuals
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.2)
        
    def compute_feats(self,y_context : List[torch.Tensor],indices):
        final1,final2,final3 = [],[],[]
        for i,embed in enumerate(self.transformer_embeddings):
            temp = torch.cdist(embed.weight[indices[i]].unsqueeze(0),embed.weight.unsqueeze(0),p=2.0)[0]+1e-3
            temp[torch.arange(temp.shape[0]),indices[i]] = 0
            temp_w = torch.exp(-temp/self.tau)
            temp_i = torch.argsort(temp_w)[:,-(self.k+1):-1]
            temp_s = y_context[i][torch.arange(y_context[i].shape[0])[:,None],:,temp_i]
            temp_w = temp_w[torch.arange(temp_w.shape[0])[:,None],None,temp_i].repeat(1,1,y_context[0].shape[1])
            temp_w[temp_s==0] = temp_w[temp_s==0]/1e3
            final1.append(temp_w.sum(dim=1,keepdim=True))
            final2.append((temp_w*temp_s).sum(dim=1,keepdim=True)/temp_w.sum(dim=1,keepdim=True))
            final3.append(torch.std(temp_s,dim=1,keepdim=True))

        return torch.cat(final1+final2+final3,dim=1).transpose(1,2)
    
    # def core(self,in_series,mask,residuals,context_info : List[torch.Tensor]):
    #     embeddings_forward = [self.conv_forward(self.pad_forward(in_series.unsqueeze(1)))]
    #     embeddings_backward = [self.conv_backward(self.pad_backward(in_series.unsqueeze(1)))]
    #     query_forward = [torch.zeros(embeddings_forward[0].shape).to(embeddings_forward[0].device)]
    #     query_backward = [torch.zeros(embeddings_backward[0].shape).to(embeddings_forward[0].device)]
    #     embedded_feats = []
    #     for i,embed in enumerate(self.transformer_embeddings):
    #         temp = embed.weight[context_info[0][:,i]]
    #         embedded_feats.append(temp.unsqueeze(2).repeat(1,1,in_series.shape[1]))
    #     temp = self.time_embeddings.weight[context_info[2].to(in_series.device)].transpose(1,2)
    #     embedded_feats.append(temp)
        
    #     embeddings_forward += embedded_feats
    #     embeddings_backward += embedded_feats
    #     query_forward += embedded_feats
    #     query_backward += embedded_feats
        
    #     series_forward = torch.cat(embeddings_forward,dim=1)
    #     series_backward = torch.cat(embeddings_backward,dim=1)
    #     query_forward = torch.cat(query_forward,dim=1)
    #     query_backward = torch.cat(query_backward,dim=1)
    #     hidden_state,mask = self.transformer(series_forward,series_backward,query_forward,query_backward,mask)
        
    #     temp = self.compute_feats([residuals],context_info[0].transpose(0,1))
    #     feats = torch.cat([temp,hidden_state],dim=2)
    #     feats = self.outlier_layer1(feats).clamp(min=0)
    #     feats = self.outlier_layer2(feats).clamp(min=0)
    #     mean = self.mean_outlier_layer(feats)[:,:,0]#.squeeze()
    #     return mean,mask

    def core(self,in_series,mask,residuals,context_info : List[torch.Tensor]):
        embeddings_forward = [self.conv_forward(self.pad_forward(in_series.unsqueeze(1)))]
        embeddings_backward = [self.conv_backward(self.pad_backward(in_series.unsqueeze(1)))]
        embeddings_forward[0] = torch.zeros(embeddings_forward[0].shape).cuda()
        embeddings_backward[0] = torch.zeros(embeddings_backward[0].shape).cuda()
        
        for i,embed in enumerate(self.transformer_embeddings):
            temp = embed.weight[context_info[0][:,i]]
            embeddings_forward.append(temp.unsqueeze(2).repeat(1,1,in_series.shape[1]))
            embeddings_backward.append(temp.unsqueeze(2).repeat(1,1,in_series.shape[1]))
            
        temp = self.time_embeddings.weight[context_info[2].to(in_series.device)].transpose(1,2)
        embeddings_forward.append(temp)
        embeddings_backward.append(temp)
        
        series_forward = torch.cat(embeddings_forward,dim=1)
        series_backward = torch.cat(embeddings_backward,dim=1)
        hidden_state = torch.cat([series_forward.transpose(1,2),series_backward.transpose(1,2)],dim = 2)
        
        temp = self.compute_feats([residuals],context_info[0].transpose(0,1))
        feats = self.dropout(torch.cat([temp,hidden_state],dim=2))
        feats = self.outlier_layer1(feats).clamp(min=0)
        feats = self.outlier_layer2(feats).clamp(min=0)
        mean = self.mean_outlier_layer(feats)[:,:,0]#.squeeze()
        return mean,mask

    def forward (self,in_series,mask,residuals,context_info : List[torch.Tensor]):
        mean,mask = self.core(in_series,mask,residuals,context_info)
        return {'mae':self.mae_loss(mean,in_series,mask).mean()}
    
    @torch.jit.export
    def validate(self,in_series,mask,residuals, context_info  : List[torch.Tensor]):
        mean,mask = self.core(in_series,mask,residuals,context_info)
        loss = self.mae_loss(mean,in_series,mask)
        return {'loss_values':loss,'values':mean}


    def mae_loss(self,y,y_pred,mask):
        temp = torch.abs((y_pred-y).cuda())
        loss = temp[mask>0]
        return loss


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self,feats,block_size=10,time_context=None):
        self.feats = feats.astype(np.float32)
        self.num_dim = len(self.feats.shape)
        self.time_context = time_context
        self.block_size = block_size
        # self.size = int(self.feats.shape[0]*self.feats.shape[1])
        if (self.time_context == None):
            self.size = self.feats.shape[1]
        else :
            self.size = int(self.feats.shape[0]*self.feats.shape[1]/time_context)
            
        self.normalised_feats = np.nan_to_num((self.feats-np.nanmean(self.feats,axis=0))/np.nanstd(self.feats,axis=0))
        
    def __getitem__(self,index):
        if (self.time_context != None):
            # time_ = (index%self.feats.shape[0])
            # tsNumber = int(index/self.feats.shape[0])
            time_ = (index%(self.feats.shape[0]*self.time_context))
            tsNumber = int(index*self.time_context/self.feats.shape[0])
            lower_limit = min(time_,self.time_context)
            series = self.feats[time_-lower_limit:time_+self.time_context]
            residuals = self.normalised_feats[time_-lower_limit:time_+self.time_context,:]
            time_vector = torch.arange(series.shape[0])+(time_-lower_limit)
        else :
            # tsNumber = int(index/self.feats.shape[0])
            tsNumber = int(index)
            series = self.feats
            residuals = self.normalised_feats
            time_vector = torch.arange(series.shape[0])
            

        series = series[:,tsNumber]
        series = copy.deepcopy(series)
        series[np.random.randint(0,series.shape[0],size=int(series.shape[0]/100))] = np.nan
        
        mask = np.ones(series.shape)
        mask[np.isnan(series)] = 0

        if (np.isnan(np.nanmean(series))):
            print (time_)
            print (tsNumber)
            print (series)
            exit()
        series = np.nan_to_num(series,nan=np.nanmean(series))

        mean = np.random.rand(1)[0]/10
        series += mean
        context = [tsNumber]
        return torch.FloatTensor(series),torch.BoolTensor(mask>0),context,torch.FloatTensor(residuals),0,time_vector
        
    def __len__(self):
        return self.size


class ValidationTransformerDataset(torch.utils.data.Dataset):
    def __init__(self,validation_feats,examples_file,block_size,test,time_context=None):
        self.validation_feats = validation_feats.astype(np.float32)
        self.examples_file = examples_file.astype(np.int)
        self.time_context = time_context
        self.normalised_feats = np.nan_to_num((self.validation_feats-np.nanmean(self.validation_feats,axis=0))/np.nanstd(self.validation_feats,axis=0))
        self.test = test
        
    def __getitem__(self,index):
        this_example = self.examples_file[index]
        time_ = this_example[0]

        if (self.time_context != None):
            lower_limit = min(time_,self.time_context)
            out_series = self.validation_feats[time_-lower_limit:time_+self.time_context+this_example[2]]
            residuals = self.normalised_feats[time_-lower_limit:time_+self.time_context+this_example[2],:]
            start_time = time_-lower_limit
            time_ = lower_limit
            time_vector = torch.arange(out_series.shape[0])+start_time
        else :
            out_series = self.validation_feats
            residuals = self.normalised_feats
            start_time = 0
            time_vector = torch.arange(out_series.shape[0])

        out_series = out_series[:,this_example[1]]
        
        upper_limit = this_example[2]
        
        mask = np.zeros(out_series.shape)
        mask[time_:time_+upper_limit] = 1
        if (not self.test):
            mask[np.isnan(out_series)] = 0
        out_series = np.nan_to_num(out_series,nan=np.nanmean(out_series))

        context = [this_example[1]]
        return torch.FloatTensor(out_series),torch.BoolTensor(mask>0),context,torch.FloatTensor(residuals),start_time,time_vector
    
        
    def __len__(self):
        return self.examples_file.shape[0]
        

def transformer_collate(batch):
    #batch = list(filter(None, batch)) 
    (series,mask,index,residuals,start_time,time_vector) = zip(*batch)
    return pad_sequence(series,batch_first=True),pad_sequence(mask,batch_first=True),pad_sequence(residuals,batch_first=True),\
        [torch.LongTensor(list(index)),torch.LongTensor(list(start_time)),pad_sequence(time_vector,batch_first=True)]

# def get_block_length(matrix):
#     num_missing = len(np.where(np.isnan(matrix))[0])
#     num_blocks = 0
#     for j in range(matrix.shape[1]):
#         temp = matrix[:,j]
#         for i in range(len(temp)-1):
#             if (np.isnan(temp[i]) and ~np.isnan(temp[i+1])):
#                 num_blocks += 1
#         if (np.isnan(temp[-1])):
#             num_blocks += 1
#     #num_blocks *= matrix.shape[1]
#     return int(num_missing/num_blocks)

def make_validation (matrix,block_size):
    validation_points = np.random.uniform(0,matrix.shape[0]-block_size,(matrix.shape[1])).astype(np.int)
    train_matrix = copy.deepcopy(matrix)
    val_points = []
    test_points = []
    for i,x in enumerate(validation_points) :
        train_matrix[x:x+block_size,i] = np.nan
        val_points.append([x,i,block_size])
    for i in range(matrix.shape[1]):
        j =0
        while j < matrix.shape[0]:
            if (np.isnan(matrix[j][i])):
                time = 0
                while j < matrix.shape[0] and np.isnan(matrix[j][i]):
                    time+= 1
                    j += 1
                test_points.append([j-time,i,time])
            else :
                j += 1
    return train_matrix,matrix,np.array(val_points),np.array(test_points)

def train(model,train_loader,val_loader,device):
    best_state_dict = model.state_dict()
    best_loss = float('inf')

    lr = 1e-2
    optim = torch.optim.Adam(model.parameters(),lr=lr)

    iteration = 0
    start_epoch = 0
    tolerance_epoch = 0
    patience  = 5
    max_epoch = 1000
    #max_epoch = 10
    for epoch in range(start_epoch,max_epoch):
        print ("Starting Epoch : %d"%epoch)

        for inp_,mask,residuals,context_info in train_loader :
            inp_ = inp_.to(device).requires_grad_(True)
            loss = model(inp_,mask.to(device),residuals.to(device),context_info)
            optim.zero_grad()
            #print (epoch,loss['mae'],best_loss)
            loss['mae'].backward()
            optim.step()
            iteration += 1
        if (epoch % 5 == 0):
            model.eval()
            loss_mre_num,count = 0,0
            with torch.no_grad():
                for inp_,mask,residuals,context_info in val_loader :
                    loss = model.validate(inp_.to(device),mask.to(device),
                                          residuals.to(device),context_info)
                    loss_mre_num += loss['loss_values'].sum()
                    count += len(loss['loss_values'])
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

def test(model,test_loader,val_feats,device):
    output_matrix = copy.deepcopy(val_feats)
    model.eval()
    with torch.no_grad():
        for inp_,mask,residuals,context_info in test_loader :
            loss = model.validate(inp_.to(device),mask.to(device),residuals.to(device),context_info)
            output_matrix[context_info[1][0]:context_info[1][0]+mask.shape[1],context_info[0][0,0]] = \
            np.where(mask.detach().cpu().numpy()[0],loss['values'].detach().cpu().numpy()[0],output_matrix[context_info[1][0]:context_info[1][0]+mask.shape[1],context_info[0][0,0]])
    model.train()
    return output_matrix


def transformer_recovery(input_feats):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device('cuda:%d'%0)
    batch_size = 1024

    block_size = 10#get_block_length(input_feats)
    print (input_feats.shape,block_size)
    train_feats,val_feats,val_points,test_points = make_validation(input_feats,block_size)
    if (input_feats.shape[0]<100):
        time_context = None
    else :
        time_context = 50
        
    time_context = None
    batch_size = train_feats.shape[1]
    train_set = TransformerDataset(train_feats,block_size,time_context = time_context)
    val_set = ValidationTransformerDataset(val_feats,val_points,block_size,False,time_context = time_context)
    test_set = ValidationTransformerDataset(val_feats,test_points,block_size,True,time_context = time_context)

    print (len(train_set))
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size = batch_size,drop_last = False,shuffle=True,collate_fn = transformer_collate)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size = 1,drop_last = False,shuffle=True,collate_fn = transformer_collate)

    # drift10,airq
    model = OurModel(sizes=[train_feats.shape[1]],nkernel=8,embedding_size=8,time_embed=16,nhid=16,nlayers=1,nhead=2,time_len=train_feats.shape[0]).to(device)
    # chlorine
    # model = OurModel(sizes=[train_feats.shape[1]],nkernel=8,embedding_size=8,time_embed=8,nhid=16,nlayers=1,nhead=2,time_len=train_feats.shape[0]).to(device)
    model.transformer.mask_len = block_size
    model = torch.jit.script(model)

    best_state_dict = train(model,train_loader,val_loader,device)
    model.load_state_dict(best_state_dict)

    matrix = test(model,test_loader,val_feats,device)
#    print (np.where(np.isnan(matrix)))
    return matrix
