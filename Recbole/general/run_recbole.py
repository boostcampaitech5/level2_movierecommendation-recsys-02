import os
import json
import argparse
import pandas as pd
import numpy as np
import time, datetime
from tqdm import tqdm
from args import parse_args
from logging import getLogger
import torch
import pdb
from util import load_data_file, save_atomic_file , make_config

from recbole.model.general_recommender.multivae import MultiVAE
from recbole.quick_start import run_recbole

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color

# 사용법
# python run_recbole.py --model_name=[] --epochs=[]
SEED=13

seq_models = ['SASRec','GRU4Rec']
general_models = ['EASE','MultiVAE','MultiDAE','ADMMSLIM','NGCF','RecVAE','FM']
context_models = ['FM','FFM','DeepFM']



def main(args):
    """모델 train 파일
    args:
        model_name(default - "MultiVAE") : 모델의 이름을 입력받습니다.
        나머지는 hyper parameter 입니다. 
    
    """
    config_name = args.config
    model_name = args.model_name
    top_k = args.top_k
    
    train_data, user_data, item_data = load_data_file()

    save_atomic_file(train_data, user_data, item_data)
    
    # config 파일이 없을 경우 생성                
    if not os.path.isfile(f'./{config_name}'):
        print("Make config...")
        make_config(config_name)
        
    parameter_dict = args.__dict__
   
    # Default eval_args를 저장
    if model_name in general_models:
        parameter_dict['eval_args'] = {
            'group_by': 'user',
            'order': 'RO',
            'mode': 'full',}
        
    # context 모델일 경우 eval_args 변경
    elif model_name in context_models:
        parameter_dict['eval_args'] = {
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full',}        
    
    # Sequential 모델일 경우 eval_args와 loss_type을 변경
    elif model_name in seq_models:
        parameter_dict['eval_args'] = {
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full',}
        parameter_dict['loss_type'] = 'BPR'
    
    #run_recbole
    print(f"running {model_name}...")
    result = run_recbole(
        model = model_name,
        dataset = 'train_data',
        config_file_list = [config_name],
        config_dict = parameter_dict,
    )
    
if __name__ == "__main__":
    args = parse_args()
    main(args)