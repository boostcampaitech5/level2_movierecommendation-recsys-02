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
"--K", default=10, : 각 유저별로 생성되는 추천의 수 
"--option", default="prior"
    - prior : 우선순위를 기준으로 앙상블
    - weight : 가중치를 부여한 앙상블

    
사용방법
1. hard voting
    ※ 우선순위로 줄 파일에 1_붙여서 돌리기
    ex) EASE_@.csv , SASRec_@.csv, Ease 우선순위로 앙상블 하는 경우 
        1_EASE_@.csv , SASRec_@.csv 로 변경해서 앙상블 돌리기
    python ensemble.py --pick {ex:@} 
2. wight sum 
    python ensemble.py --pick {ex:@} --option weight 명령어 입력
    -> ratio 입력 창 뜸 -> 띄어쓰기해서 ratio 입력하기
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pick", default=None, type=str)
    #parser.add_argument("--ratio", default=None, type=list)
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


def ensemble_prior(file_list,args):
    dataframe_dict ={} 
    for i in range(len(file_list)):
        file = file_list[i]
        
        if file.split("/")[-1][0] == "1":
            prior1_df = pd.read_csv(file)
            prior1_df = prior1_df.groupby('user').head(8)
            prior1_df['prior'] = [i for i in range(8)]*31360
            dataframe_dict["1"] = prior1_df
        else:
            prior2_df = pd.read_csv(file)
            prior2_df['prior'] = [i for i in range(8,18)]*31360
            dataframe_dict["2"] = prior2_df

    result = pd.concat([dataframe_dict["1"],dataframe_dict["2"]])
    result = result.drop_duplicates(['user','item'],keep='first')
    result = result.sort_values(['user','prior']).reset_index(drop=True)
    result = result.groupby('user').head(10)
    return result[['user','item']]

#hard voting
def ensemble_weight(file_list):
    #if option == 'prior':
    ratios = input(f"(모델 수 : {len(file_list)})띄어쓰기로 구분하여 ratio 입력하기") 
    ratios = ratios.split(" ")
    ratios = list(map(int,ratios))
    if len(ratios) != len(file_list):
        raise ValueError("모델 수와 가중치값의 수가 맞지 않음") 
    df_list = [pd.read_csv(f) for f in file_list]

    for df,ratio in zip(df_list,ratios):
        df["weight"] = ratio
    result = pd.concat(df_list).reset_index(drop=True)
    g = result.groupby(['user','item']).agg({'weight': 'sum'})
    g['weight'] = - g['weight']
    g = g.sort_values(['user','weight'])
    g['weight'] = - g['weight']
    result = g.groupby('user').head(10)
    result = result.reset_index()
    return result[['user','item']] , ratios

#weight
def make_csv(args,file_list):
    option = args.option
    if option == "prior":
        result = ensemble_prior(file_list,args)
        filename= f"{args.option} : "+ "_".join([f"{m.split('/')[-1][:-4]}" for m in file_list if ".csv" in m])
        result.to_csv(f'./output/{filename}.csv',index=False)
    
    elif option == "weight":
        result,ratios = ensemble_weight(file_list)
        filename= f"{args.option} _ "+ "+".join([f"{m.split('/')[-1][:-4]}_{r}" for m,r in zip(file_list,ratios) if ".csv" in m])
        result.to_csv(f'./output/{filename}.csv',index=False)


if __name__ == "__main__":
    args = parse_args()
    file_list = check_attribute(args)
    make_csv(args,file_list)