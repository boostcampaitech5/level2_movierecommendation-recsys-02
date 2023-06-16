import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import glob
import pdb
'''
인자설정
"--pick", default=None : output 폴더에서 앙상블할 파일에 공통으로 들어가는 단어 입력
"--ratio", default=None, type=list : 
        weight 앙상블의 경우 웨이
"--K", default=10, : 각 유저별로 생성되는 추천의 수 
"--option", default="prior"
    - prior : 우선순위를 기준으로 앙상블
    - weight : 가중치를 부여한 앙상블

    
사용방법
1. python ensemble.py 
2. 앙상블할 ratio 입력하기.(앙상블할 모델의 수만큼 입력해줘야 함) ex) 111  
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default=None, type=str)
    parser.add_argument("--ratio", default=None, type=list)
    parser.add_argument("--K", default=10, type=int)
    parser.add_argument("--option", default="prior", type=str ,help="[prior,weight]")

    return parser.parse_args()

# 앙상블 속성값 확인 및 파일 목록 생성
def check_attribute(args):
    
    file_list = glob.glob('./output/*.csv')
    
    if args.pick is not None:
        file_list = [i for i in file_list if args.pick in i]

    if len(file_list) <= 1:
        raise ValueError("앙상블할 모델을 2개 이상 입력해주세요.")
        
    print("앙상블할 모델 개수 :",len(file_list),end="\t")
    print("앙상블 방식",args.option)
    print("앙상블할 파일 목록 :",file_list)

    return file_list


def ensemble_prior():
    dataframe_list = []
    prior = []
    for i in range(len(file_list)):
        file = file_list[i]
        if file[0] == "1":
            prior_df = pd.read_csv(file_list[i])
            prior_df.groupby('user').head(8)
            dataframe_list.append(temp_df)
        else:
            temp_df = pd.read_csv(file_list[i])
            dataframe_list.append(temp_df)

    
        temp_df['prior'] = [k for k in range((i+1)*8)]*31360

        
        result = pd.concat(dataframe_dict)
        result = result.drop_duplicates(['user','item'],keep='first')
        result = result.sort_values(['user','prior'])
        result = result.groupby('user').head(10)

    return result[['user','item']]

def ensemble_weight():

    W_dict = dict(zip(file_list,ratio))
    #print("딕셔너리 확인 : ",W_dict)
    ratios = list(W_dict.values())
    ratios = [float(ratio) for ratio in ratios]
    return None

def make_csv(file_list,args):
    W_dict = dict(zip(file_list,ratio))
    #print("딕셔너리 확인 : ",W_dict)
    ratios = list(W_dict.values())
    ratios = [float(ratio) for ratio in ratios]

    if option == "prior":
        result = ensemble_prior()
    elif option == "weight":
        result = ensemble_weight()

    filename="*".join([f"{args.option}:{m.split('/')[-1][:-4]} : {r}" for m,r in zip(file_list,ratios) if ".csv" in m])
    result.to_csv(f'./output/{filename}.csv',index=False)


if __name__ == "__main__":
    args = parse_args()
    file_list = file_list(args)
    #pdb.set_trace()
    make_csv(file_list,args)