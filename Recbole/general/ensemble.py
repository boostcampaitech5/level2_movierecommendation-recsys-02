import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import glob

def pick_model(args):
    pick_model = glob.glob('*.csv')
    if args.pick is not None:
        pick_model = [i for i in pick_model if args.pick in i]
    print(pick_model)
    return pick_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default=None, type=str)
    parser.add_argument("--ratio", default=None, type=list)
    return parser.parse_args()

def make_csv(models_list,ratio):
    W_dict = dict(zip(models_list,ratio))
    print("model_list",W_dict)

    dataframe_list = []

    for i in range(len(models_list)):
        dataframe_list.append(pd.read_csv(models_list[i]))


if __name__ == "__main__":
    args = parse_args()
    models_list = pick_model(args)
    make_csv(models_list,args.ratio)
















# dataframe_list = []

# print('순위별 가중치값 입력(ex: 1 0.9 0.8 ...)')
# rank_ratio = [1 for _ in range(10)] + [0.3 for _ in range(5)]
# rank_len = len(rank_ratio)

# print(f"앙상블 모델 개수: {len(filepaths)}")

# for i in range(len(filepaths)):
#     dataframe_list.append(pd.read_csv(filepaths[i]))

# user_list = dataframe_list[0]['user'].unique()
# dataframe_len = len(dataframe_list)


# ratios = list(w_dict.values())
# K=10
# result = []
# tbar = tqdm(user_list, desc='Ensemble')
# for user in tbar:
#     temp = defaultdict(float)
#     for df_idx in range(dataframe_len):
#         items = dataframe_list[df_idx][dataframe_list[df_idx]['user'] == user]['item'].values
#         max_rank = min(len(items), rank_len)
#         for rank_idx in range(max_rank):
#             temp[items[rank_idx]] += rank_ratio[rank_idx] * ratios[df_idx]

#     for key, _ in sorted(temp.items(), key=lambda x: x[1], reverse=True)[:K]:
#         result.append((user, key))