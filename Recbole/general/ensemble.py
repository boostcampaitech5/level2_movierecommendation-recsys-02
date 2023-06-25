import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import glob
import pdb
import math
import sys
import tqdm
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

    #parser.add_argument("--Top_K", default=0, type=int)
    #parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--option", default="prior", type=str ,help="[prior,weight,double]")

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
            prior2_df = prior2_df.groupby('user').head(10)
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


# # model별 랭크별 가중치를 설정하여 11-15위 추천도 활용
# def ensemble_double_ratio():
def ensemble_double_ratio(args,file_path):
    
    # ratio1 - Top1-k : Topk-15
    print("Write ratio for Top1-10 and Top11-15. Sum of ratios must be 1.\nex) 0.75 0.25")
    inputs = input().strip()
    top_weight_list = [float(item) for item in inputs.split() if inputs]

    # exception handling
    if len(top_weight_list) != 2:
        ValueError("Wrong ratios. Try again.\n")
    if math.fabs(sum(top_weight_list) - 1.0) > sys.float_info.epsilon:
        ValueError("Sum of ratios is not 1. Try again.\n")



    # ratio2 - 모델별 weight 
    print("\nWrite ratio for input csv files. Sum of ratios must be 1.\nex) 0.5 0.5")
    inputs = input().strip()
    model_weight_list = [float(item) for item in inputs.split() if inputs]

    # exception handling
    if len(model_weight_list) < 2:
        ValueError("Wrong ratios. Try again.\n")
    if math.fabs(sum(model_weight_list) - 1.0) > sys.float_info.epsilon:
        ValueError("Sum of ratios is not 1. Try again.\n")



    # summary inputs
    print("\n----------input information----------")
    print(f"Ratio of Top1-k & Topk-15:\n{top_weight_list[0]} : {top_weight_list[1]}")
    print("Ratio of input csv files:")
    for idx in range(len(file_path)):
        print(f"[{idx}] : {file_path[idx]} - {model_weight_list[idx]}")
    print("-------------------------------------\n")


    top1015 = pd.read_csv(file_path[0]).groupby("user").apply(lambda x: x.iloc[:10,1]).reset_index().drop('level_1', axis=1)
    top1015.head(10)

    print(top_weight_list,model_weight_list# top 1,2
        )

    file_list = []
    for path in file_path:
        csv = pd.read_csv(path)
        file_list.append(csv)

    user_list = file_list[0]["user"].unique()

    #file_path



    # voting
    '''
    ex)
        model A : model B = 0.6 : 0.4
        top1-10 : top11-15 = 0.75 : 0.25

    1. model A & Top1-10 = 0.6 x 0.75 ~= 0.45
    2. model A & Top10-15 = 0.6 x 0.25 ~= 0.15
    3. model B & Top1-10 = 0.4 x 0.75 = 0.3
    4. model B & Top10-15 = 0.4 x 0.25 = 0.1
    '''
    #user_list = file_list[0]["user"].unique()
    file_list = []
    for m_id, path in enumerate(file_path):

        top0110 = pd.read_csv(path).groupby("user").apply(lambda x: x.iloc[:10,1]).reset_index().drop('level_1', axis=1)
        top1115 = pd.read_csv(path).groupby("user").apply(lambda x: x.iloc[10:15,1]).reset_index().drop('level_1', axis=1)

        top0110['weight'] = model_weight_list[m_id] * top_weight_list[0]
        top1115['weight'] = model_weight_list[m_id] * top_weight_list[1]
        file_list.append(top0110)
        file_list.append(top1115)

    #get user information


    #print(len(file_list))

    result = pd.concat(file_list).reset_index(drop=True)
    #result[result['user']==11]


    g = result.groupby(['user','item']).agg({'weight': 'sum'})
    g['weight'] = - g['weight']
    g = g.sort_values(['user','weight'])
    g['weight'] = - g['weight']
    result = g.groupby('user').head(10).reset_index()

    return result[['user','item']]

#weight
def make_csv(args,file_list):
    option = args.option
    if option == "prior":
        result = ensemble_prior(file_list,args)
        filename= f"{args.option}_"+ "+".join([f"{m.split('/')[-1][:-4]}" for m in file_list if ".csv" in m])
        result.to_csv(f'./output/{filename}.csv',index=False)
    
    elif option == "weight":
        result,ratios = ensemble_weight(file_list)
        filename= f"{args.option} _ "+ "+".join([f"{m.split('/')[-1][:-4]}*{r}" for m,r in zip(file_list,ratios) if ".csv" in m])
        result.to_csv(f'./output/{filename}.csv',index=False)

    elif option == "db_weight":
        result,ratios = ensemble_weight(args,file_list)
        filename= f"{args.option} _ "+ "+".join([f"{m.split('/')[-1][:-4]}*{r}" for m,r in file_list if ".csv" in m])
        result.to_csv(f'./output/{filename}.csv',index=False)


if __name__ == "__main__":
    args = parse_args()
    file_list = check_attribute(args)
    make_csv(args,file_list)