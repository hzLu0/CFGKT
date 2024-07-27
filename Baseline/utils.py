import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import random
import json
import time, datetime
from sklearn.model_selection import train_test_split

class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def lt_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool).to(device)

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PadSequence(object):
    def __call__(self, batch: List[Tuple[torch.Tensor]]):
        batch = sorted(batch, key=lambda y: y[0].shape[0], reverse=True)

        effLen = torch.cat([x[0] for x in batch])

        currQuestionAddLabel = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
        currQuestionID = torch.nn.utils.rnn.pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0)

        currSkillAddLabel = torch.nn.utils.rnn.pad_sequence([x[3] for x in batch], batch_first=True, padding_value=0)
        currSkillID = torch.nn.utils.rnn.pad_sequence([x[4] for x in batch], batch_first=True, padding_value=0)
        currSkill_oneHot = torch.nn.utils.rnn.pad_sequence([x[5] for x in batch], batch_first=True)

        currLabel = torch.nn.utils.rnn.pad_sequence([x[6] for x in batch], batch_first=True, padding_value=0)
        nextQuestionID = torch.nn.utils.rnn.pad_sequence([x[7] for x in batch], batch_first=True, padding_value=0)

        nextSkillID = torch.nn.utils.rnn.pad_sequence([x[8] for x in batch], batch_first=True, padding_value=0)
        nextSkill_oneHot = torch.nn.utils.rnn.pad_sequence([x[9] for x in batch], batch_first=True)

        nextLabel = torch.nn.utils.rnn.pad_sequence([x[10] for x in batch], batch_first=True, padding_value=0)

        return effLen, \
               currQuestionAddLabel, currQuestionID, \
               currSkillAddLabel, currSkillID,currSkill_oneHot,\
               currLabel, \
               nextQuestionID, \
               nextSkillID, nextSkill_oneHot, \
               nextLabel

def q_c_difficulty():
    pd.set_option('display.float_format',lambda x : '%.2f' % x)
    np.set_printoptions(suppress=True)
    all_data =  pd.read_csv('data/2012-2013-data-with-predictions-4-final.csv', encoding = "ISO-8859-1", low_memory=False)
    print(all_data.head())
    order = ['user_id','problem_id','correct','skill_id']
    all_data2 = all_data[order]
    all_data2['skill_id'].fillna('nan',inplace=True)
    all_data = all_data2[all_data2['skill_id'] != 'nan'].reset_index(drop=True)
    skill_id = np.array(all_data['skill_id'])
    skills = set(skill_id)
    print('# of skills:',  len(skills))
    user_id = np.array(all_data['user_id'])
    problem_id = np.array(all_data['problem_id'])
    user = set(user_id)
    problem = set(problem_id)
    user2id ={}
    problem2id = {}
    skill2id = {}
    count = 1
    for i in user:
        user2id[i] = count 
        count += 1
    count = 1
    for i in problem:
        problem2id[i] = count 
        count += 1
    count = 0
    for i in skills:
        skill2id[i] = count 
        count += 1
    with open('data/user2id', 'w', encoding = 'utf-8') as fo:
        fo.write(str(user2id))
    with open('data/problem2id', 'w', encoding = 'utf-8') as fo:
        fo.write(str(problem2id))
    with open('data/skill2id', 'w', encoding = 'utf-8') as fo:
        fo.write(str(skill2id))
        
    # KC difficulty
    sdifficult2id = {}
    count = []
    nonesk = []   #  dropped with less than 30 answer records
    for i in tqdm(skills):
        tttt = []
        idx = all_data[(all_data.skill_id==i)].index.tolist() 
        temp1 = all_data.iloc[idx]
        if len(idx) < 30:
            sdifficult2id[i] = 1.02
            nonesk.append(i)
            continue
        for xxx in np.array(temp1):
            tttt.append(xxx[2])
        if tttt == []:

            sdifficult2id[i] = 1.02
            nonesk.append(i)
            continue
        avg = int(np.mean(tttt)*100)+1
        count.append(avg)
        sdifficult2id[i] = avg 

    # Question difficulty
    difficult2id = {}
        count = []
    nones = []
    for i in tqdm(problem):
        tttt = []
        idx = all_data[(all_data.problem_id==i)].index.tolist() 
        temp1 = all_data.iloc[idx]
        if len(idx) < 30:
            difficult2id[i] = 1.02
            nones.append(i)
            continue
        for xxx in np.array(temp1):
            tttt.append(xxx[2])
        if tttt == []:
            difficult2id[i] = 1.02
            nones.append(i)
            continue
        avg = int(np.mean(tttt)*100)+1
        count.append(avg)
        difficult2id[i] = avg 

    with open('data/difficult2id', 'w', encoding = 'utf-8') as fo:
        fo.write(str(difficult2id))
    with open('data/sdifficult2id', 'w', encoding = 'utf-8') as fo:
        fo.write(str(sdifficult2id))
    np.save('data/nones.npy', np.array(nones))
    np.save('data/nonesk.npy', np.array(nonesk))
    return sdifficult2id, difficult2id



 
