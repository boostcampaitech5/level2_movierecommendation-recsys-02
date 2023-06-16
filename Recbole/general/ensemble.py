import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import glob
'''
사용방법
1. python ensemble.py 
2. 앙상블할 ratio 입력하기.(앙상블할 모델의 수만큼 입력해줘야 함) ex) 111  
'''


def models_list(args):
    models_list = glob.glob('*.csv')
    if args.pick is not None:
        models_list = [i for i in models_list if args.pick in i]
    print("앙상블 모델 개수 :",len(models_list))
    print("앙상블 모델 목록 :",models_list)
    return models_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default=None, type=str)
    parser.add_argument("--ratio", default=input("앙상블할 ratio 를 입력하시오."), type=list)
    parser.add_argument("--K", default=10, type=int)

    
    return parser.parse_args()

def make_csv(models_list,ratio,K):
    W_dict = dict(zip(models_list,ratio))

    
    print("딕셔너리 확인 : ",W_dict)
    ratios = list(W_dict.values())
    ratios = [float(ratio) for ratio in ratios]


    dataframe_list = []

    for i in range(len(models_list)):
        dataframe_list.append(pd.read_csv(models_list[i]))

    K=K
    result = []
    user_list = dataframe_list[0]['user'].unique()
    tbar = tqdm(user_list, desc='Ensemble')
    dataframe_len = len(dataframe_list)
    rank_len = len(W_dict)
    rank_ratio = list(map(int, ratio))
    for user in tbar:
        temp = defaultdict(float)
        for df_idx in range(dataframe_len):
            items = dataframe_list[df_idx][dataframe_list[df_idx]['user'] == user]['item'].values
            max_rank = min(len(items), rank_len)
            for rank_idx in range(max_rank):
                temp[items[rank_idx]] += rank_ratio[rank_idx] * ratios[df_idx]

        for key, _ in sorted(temp.items(), key=lambda x: x[1], reverse=True)[:K]:
            result.append((user, key))
    
    filename="*".join([f"{m.split('.')[0]} : {r}" for m,r in zip(models_list,ratios) if ".csv" in m])
    submission = pd.DataFrame(result, columns=['user', 'item'])
    submission.to_csv(f'{filename}.csv', index=False)
        

if __name__ == "__main__":
    args = parse_args()
    models_list = models_list(args)
    make_csv(models_list,args.ratio,args.K)