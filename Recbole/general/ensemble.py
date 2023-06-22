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
#     # ratio1 - Top1-10 : Top11-15
#     print("Write ratio for Top1-10 and Top11-15. Sum of ratios must be 1.\nex) 0.75 0.25")
#     inputs = input().strip()
#     ratio1_list = [float(item) for item in inputs.split() if inputs]

#     # exception handling
#     if len(ratio1_list) != 2:
#         ValueError("Wrong ratios. Try again.\n")
#         continue
#     if math.fabs(sum(ratio1_list) - 1.0) > sys.float_info.epsilon:
#         ValueError("Sum of ratios is not 1. Try again.\n")

#     # ratio2 - 모델별 weight 
#     print("\nWrite ratio for input csv files. Sum of ratios must be 1.\nex) 0.5 0.5")
#     inputs = input().strip()
#     ratio2_list = [float(item) for item in inputs.split() if inputs]

#     # exception handling
#     if len(ratio2_list) < 2:
#         ValueError("Wrong ratios. Try again.\n")
#     if math.fabs(sum(ratio2_list) - 1.0) > sys.float_info.epsilon:
#         ValueError("Sum of ratios is not 1. Try again.\n")

#     # summary inputs
#     print("\n----------input information----------")
#     print(f"Ratio of Top1-10 & Top11-15:\n{ratio1_list[0]} : {ratio1_list[1]}")
#     print("Ratio of input csv files:")
#     for idx in range(len(file_list)):
#         print(f"[{idx}] : {file_list[idx]} - {ratio2_list[idx]}")
#     print("-------------------------------------\n")

#     # get user information
#     user_list = file_list[0]["user"].unique()


#     # voting
#     '''
#     ex)
#       model A : model B = 0.6 : 0.4
#       top1-10 : top11-15 = 0.75 : 0.25

#     1. model A & Top1-10 = 0.6 x 0.75 ~= 0.45
#     2. model A & Top10-15 = 0.6 x 0.25 ~= 0.15
#     3. model B & Top1-10 = 0.4 x 0.75 = 0.3
#     4. model B & Top10-15 = 0.4 x 0.25 = 0.1
#     '''
#     movie_list = []
#     idx = 0
#     for user in tqdm(user_list, desc="Voting"):
#         tmp_dict = dict()
#         for i, csv in enumerate(file_list):
#             for add in range(10):
#                 if csv["item"][idx + add] not in tmp_dict:
#                     tmp_dict[csv["item"][idx + add]] = ratio1_list[0] * ratio2_list[i]
#                 else:
#                     tmp_dict[csv["item"][idx + add]] += ratio1_list[0] * ratio2_list[i]
#             for add in range(10, 15):
#                 if csv["item"][idx + add] not in tmp_dict:
#                     tmp_dict[csv["item"][idx + add]] = ratio1_list[1] * ratio2_list[i]
#                 else:
#                     tmp_dict[csv["item"][idx + add]] += ratio1_list[1] * ratio2_list[i]
#         sorted_dict = sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True)
#         for i in range(10):
#             movie_list.append(sorted_dict[i][0])
#         idx += 15

    


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


if __name__ == "__main__":
    args = parse_args()
    file_list = check_attribute(args)
    make_csv(args,file_list)