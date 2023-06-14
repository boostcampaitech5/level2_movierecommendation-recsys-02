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
# import wandb
from util import load_data_file, save_atomic_file , make_config

from recbole.model.general_recommender.multivae import MultiVAE
from recbole.quick_start import run_recbole

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color

from util_yaml import load_yaml
import pdb

# 사용법
# python run_recbole.py --model_name=[] --epochs=[]
SEED=13

seq_models = ['SASRec','GRU4Rec']
general_models = ['EASE','MultiVAE','MultiDAE','ADMMSLIM','NGCF','RecVAE','FM']
context_models = ['FM','FFM','DeepFM']


def run(args):
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    if args.model_name in [
        "MultiVAE",
        "MultiDAE",
        "RecVAE",
        "EASE"

    ]:
        parameter_dict = {
            "neg_sampling": None,
        }

        return run_recbole(
            model=args.model_name,
            dataset='train_data',
            config_file_list=[os.path.join(curr_dir, 'yaml_dir', f'{args.model_name}.yaml')],
            config_dict=parameter_dict,
        )
    else:
        return run_recbole(
            model=args.model_name,
            dataset='train_data',
            config_file_list=[os.path.join(curr_dir, 'yaml_dir', f'{args.model_name}.yaml')],
        )

def main(args):
    """모델 train 파일
    args:
        model_name(default - "MultiVAE") : 모델의 이름을 입력받습니다.
        나머지는 hyper parameter 입니다. 
    
    """
    #wandb.login()
    #wandb.init(project='movierec', entity='recommy_movierec')
    #wandb.run.name = f'{model_name}_{config_name}_epoch{args.epochs}'   
    
    train_data, user_data, item_data = load_data_file()

    save_atomic_file(train_data, user_data, item_data)
            
    load_yaml(args)
    # run
    print(f"running {args.model_name}...")
    start = time.time()
    result = run(args)
    t = time.time() - start
    print(f"It took {t/60:.2f} mins")
    print(result)
    
    #wandb.run.finish()
if __name__ == "__main__":
    args = parse_args()
    main(args)