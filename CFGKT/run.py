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


def run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtypes = {'uuid': 'int32',
              "upid": "int16", 'ucid': 'int16','is_correct': 'int8'}
    train_df = pd.read_csv('./New_Log_Problem.csv', usecols=['uuid', 'upid', 'ucid', 'is_correct', 'timestamp_TW',
                                                             'total_sec_taken'], dtype=dtypes, encoding='gbk')
    print("shape of dataframe :", train_df.shape)
    raw_skill = train_df.ucid.unique().tolist()
    raw_problem = train_df.upid.unique().tolist()
    sub_skills = {p: i for i, p in enumerate(raw_skill)}
    sub_problems = {p: i for i, p in enumerate(raw_problem)}
    train_df['ucid'] = train_df['ucid'].map(sub_skills)
    train_df['upid'] = train_df['upid'].map(sub_problems)
    skills = train_df.upid.unique()
    n_skills = len(skills)
    n_cats = len(train_df.ucid.unique())
    print("no. of skills :", n_skills)
    print("no. of categories: ", n_cats)
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
    train_dataset = CFGKT_dataset(train.values, n_skills=n_skills, n_concept=n_cats, max_seq=args.max_len)
    val_dataset = CFGKT_dataset(val.values, n_skills=n_skills, n_concept=n_cats, max_seq=args.max_len)
    test_dataset = CFGKT_dataset(test.values, n_skills=n_skills, n_concept=n_cats, max_seq=args.max_len)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=2,
                              shuffle=True,
                              worker_init_fn=np.random.seed(seed=300))
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=2,
                            shuffle=False,
                            worker_init_fn=np.random.seed(seed=300))
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=2,
                             shuffle=False,
                             worker_init_fn=np.random.seed(seed=300))

    del train_dataset, val_dataset, test_dataset
    gc.collect()
    sakt = CFGKT(n_question=n_cats, n_pid=n_skills, d_model=args.embed_dim, n_blocks=args.n_blocks,
                 kq_same = args.kq_same, dropout=args.dropout, model_type='CFGKT',memory_size=args.memory_size,
               final_fc_dim=args.final_fc_dim,n_heads=args.n_heads,d_ff=args.d_ff,
               time=args.time,interval=args.interval)
    optimizer = torch.optim.Adam(sakt.parameters(), lr=args.learning_rate)
    sakt_loss = Loss()
    train_one_epoch(sakt, train_loader, val_loader,optimizer, sakt_loss, args.epoch,device)
    save_model = torch.load('./best_model.pth')
    test_one_epoch(model=save_model, test_iterator=test_loader, device=device)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="train SAKT_ednet")
    arg_parser.add_argument("--learning_rate",
                            dest="learning_rate",
                            default=0.001,
                            type=float,
                            required=False)
    arg_parser.add_argument("--kq_same",
                            dest="kq_same",
                            default=1,
                            type=int,
                            required=False)
    arg_parser.add_argument("--n_blocks",
                            dest="n_blocks",
                            default=4,
                            type=int,
                            required=False)
    arg_parser.add_argument("--memory_size",
                            dest="memory_size",
                            default=60,
                            type=int,
                            required=False)
    arg_parser.add_argument("--batch_size",
                            dest="batch_size",
                            default=32,
                            type=int,
                            required=False)
    arg_parser.add_argument("--time",
                            dest="time",
                            default=300,
                            type=int,
                            required=False)
    arg_parser.add_argument("--interval",
                            dest="interval",
                            default=1440,
                            type=int,
                            required=False)
    arg_parser.add_argument("--final_fc_dim",
                            dest="final_fc_dim",
                            default=512,
                            type=int,
                            required=False)
    arg_parser.add_argument("--n_heads",
                            dest="n_heads",
                            default=8,
                            type=int,
                            required=False)
    arg_parser.add_argument("--d_ff",
                            dest="d_ff",
                            default=1024,
                            type=int,
                            required=False)
    arg_parser.add_argument("--embed_dim",
                            dest="embed_dim",
                            default=256,
                            type=int,
                            required=False)
    arg_parser.add_argument("--dropout",
                            dest="dropout",
                            default=0.05,
                            type=float,
                            required=False)
    arg_parser.add_argument("--epoch",
                            dest="epoch",
                            default=100,  # 15
                            type=int,
                            required=False)
    arg_parser.add_argument("--max_len",
                            dest="max_len",
                            default=100,
                            type=int,
                            required=False)
    arg_parser.add_argument("--save_params",
                            dest="save_params",
                            default=False,
                            type=bool,
                            required=False)
    args = arg_parser.parse_args()

    run(args)
