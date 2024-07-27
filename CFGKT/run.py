from prepare_data import CFGKT_dataset
from CFGKT import CFGKT
from Loss import Loss
from utils import train_one_epoch, test_one_epoch
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
import numpy as np
from config import set_opt

def run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtypes = {'uuid': 'int32',
              "upid": "int16", 'ucid': 'int16','is_correct': 'int8'}
    train_df = pd.read_csv('./Junyi.csv', usecols=['uuid', 'upid', 'ucid', 'is_correct', 'timestamp_TW',
                                                             'total_sec_taken'], dtype=dtypes, encoding='gbk')
    print("shape of dataframe :", train_df.shape)
    raw_skill = train_df.ucid.unique().tolist()
    raw_problem = train_df.upid.unique().tolist()
    sub_skills = {p: i for i, p in enumerate(raw_skill)}
    sub_problems = {p: i for i, p in enumerate(raw_problem)}
    train_df['ucid'] = train_df['ucid'].map(sub_skills)
    train_df['upid'] = train_df['upid'].map(sub_problems)
    n_ques = len(train_df.upid.unique())
    n_conc = len(train_df.ucid.unique())
    print("no. of questions :", n_ques)
    print("no. of concepts: ", n_conc)
    print("shape after exlusion:", train_df.shape)

    train_df['total_sec_taken'] = train_df['total_sec_taken'].clip(0, 300)
    train_df['timestamp_TW'] = train_df['timestamp_TW'].str[:-4]
    train_df['timestamp_TW'] = pd.to_datetime(train_df['timestamp_TW'])
    train_df['timestamp_TW'] = train_df['timestamp_TW'].astype(int) / (10 ** 9)
    train_df['timestamp_TW'] = (train_df['timestamp_TW'] / 60)

    group = train_df[["uuid", "upid", "ucid", "is_correct",'timestamp_TW','total_sec_taken']].groupby("uuid").apply(
        lambda r: (r.upid.values, r.ucid.values, r.is_correct.values,r.timestamp_TW.values,r.total_sec_taken.values))
    del train_df
    gc.collect()
    print("splitting")
    train, val = train_test_split(group, test_size=0.2)
    val, test = train_test_split(val,test_size=0.5)
    print("train size: ", train.shape, "validation size: ", val.shape, "Test size: ",test.shape)
    train_dataset = CFGKT_dataset(train.values, n_ques=n_ques, n_concept=n_conc, max_seq=args.max_len)
    val_dataset = CFGKT_dataset(val.values, n_ques=n_ques, n_concept=n_conc, max_seq=args.max_len)
    test_dataset = CFGKT_dataset(test.values, n_ques=n_ques, n_concept=n_conc, max_seq=args.max_len)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    del train_dataset, val_dataset, test_dataset
    gc.collect()
    KT_model = CFGKT(n_concepts=n_conc, n_pid=n_ques, d_model=args.embed_dim, n_blocks=args.n_blocks,
                 kq_same = args.kq_same, dropout=args.dropout, model_type='CFGKT',memory_size=args.memory_size,
               final_fc_dim=args.final_fc_dim,n_heads=args.n_heads,d_ff=args.d_ff,
               time=args.time,interval=args.interval)
    optimizer = torch.optim.Adam(sakt.parameters(), lr=args.learning_rate)
    kt_loss = Loss()
    train_one_epoch(KT_model, train_loader, val_loader,optimizer, kt_loss, args.epoch,device)
    save_model = torch.load('./best_model.pth')
    test_one_epoch(model=save_model, test_iterator=test_loader, device=device)

if __name__ == "__main__":
    dataset_name = 'Junyi'
    args = set_opt(dataset_name)
    run(args)
