import torch
import numpy as np
import argparse
import torch.nn as nn
import os
import _pickle as cPickle
import random
import math,copy
import torch.nn.functional as F
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast

from torch import nn
from einops import rearrange, repeat

from functools import partial
from contextlib import contextmanager

@contextmanager
def null_context():
    yield

def is_blackout(matrix):
    arr = (np.sum(np.isnan(matrix).astype(np.int),axis=1) == matrix.shape[1])
    return arr.astype(np.int).sum() > 0


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

# def get_block_length(matrix):
#     temp = np.where(np.isnan(matrix))
#     time = temp[0][0]
#     ts = temp[1][0]
#     i = 0
#     while (np.isnan(matrix[time+i,ts])):
#         i += 1
#     return i

# def get_block_length(matrix):
#     tss = np.unique(np.where(np.isnan(matrix))[1])
#     block_size = float('inf')
#     for ts in tss:
#         time = np.where(np.isnan(matrix[:,ts]))[0][0]
#         i = 0
#         while (time+i < matrix.shape[0] and np.isnan(matrix[time+i,ts])):
#             i += 1
#         block_size = min(block_size,i)
#     return int(block_size)

def make_validation (matrix,num_missing=20):
    np.random.seed(0)
    nan_mask = np.isnan(matrix)
    padded_mat = np.concatenate([np.zeros((1,nan_mask.shape[1])),nan_mask,np.zeros((1,nan_mask.shape[1]))],axis=0)
    indicator_mat = (padded_mat[1:,:]-padded_mat[:-1,:]).T
    pos_start = np.where(indicator_mat==1)
    pos_end = np.where(indicator_mat==-1)
    lens = (pos_end[1]-pos_start[1])[:,None]
    start_index = pos_start[1][:,None]
    time_series = pos_start[0][:,None]
    test_points = np.concatenate([start_index,time_series,lens],axis=1)
    temp = np.copy(test_points[:,2])
    if (temp.shape[0]>1):
        block_size = temp[int(temp.shape[0]/10):-int(temp.shape[0]/10)-1].mean()
    else :
        block_size = temp.mean()
    w = int(10*np.log10(block_size))
    val_block_size = int(min(block_size,w))
    num_missing = int(num_missing/val_block_size)
    train_matrix = copy.deepcopy(matrix)
    val_points = []
    
    for _ in range(num_missing):
        validation_points = np.random.uniform(0,matrix.shape[0]-val_block_size,(matrix.shape[1])).astype(np.int)
        for i,x in enumerate(validation_points) :
            train_matrix[x:x+val_block_size,i] = np.nan
            val_points.append([x,i,val_block_size])
            
    return train_matrix,matrix,np.array(val_points),test_points,int(block_size),w

    # test_possible_points = np.where(np.isnan(matrix.T))
    # i = 0
    # while i < len(test_possible_points[0]):
    #     ts_number = test_possible_points[0][i]
    #     if (test_possible_points[1][i]+block_size < matrix.shape[0] and np.isnan(matrix[test_possible_points[1][i]+block_size,ts_number])):
    #         j = block_size
    #         while (test_possible_points[1][i]+j < matrix.shape[0] and np.isnan(matrix[test_possible_points[1][i]+j,ts_number])):
    #             j += 1
    #         test_points.append([test_possible_points[1][i],ts_number,j])
    #         i += j
    #     else :
    #         test_points.append([test_possible_points[1][i],ts_number,block_size])
    #         i += block_size
    # return train_matrix,matrix,np.array(val_points),np.array(test_points)

    # for i in range(matrix.shape[1]):
    #     j =0
    #     while j < matrix.shape[0]:
    #         if (np.isnan(matrix[j][i])):
    #             time = 0
    #             while j < matrix.shape[0] and np.isnan(matrix[j][i]):
    #                 time+= 1
    #                 j += 1
    #             test_points.append([j-time,i,time])
    #         else :
    #             j += 1
    # return train_matrix,matrix,np.array(val_points),np.array(test_points)


