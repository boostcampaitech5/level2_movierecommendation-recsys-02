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

from recbole.model.general_recommender.multivae import MultiVAE
from recbole.quick_start import run_recbole

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color


SEED=13

model_list = ['MultiVAE','MultiDAE','RecVAE','EASE']

def run(args,model_name):
    if model_name in [
        "MultiVAE",
        "MultiDAE",
        "RecVAE",
        "EASE"

    ]:
        parameter_dict = {
            "neg_sampling": None,
        }
        return run_recbole(
            model=model_name,
            dataset='train_data',
            config_file_list=['general.yaml'],
            config_dict=parameter_dict,
        )
    else:
        return run_recbole(
            model=model_name,
            dataset='train_data',
            config_file_list=['general.yaml'],
        )
    
def main(args):
    """모델 train 파일
    args:
        model_name(default - "MultiVAE") : 모델의 이름을 입력받습니다.
        나머지는 hyper parameter 입니다. 
    """
    # train load
    train = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
    
    # indexing save
    user2idx = {v:k for k,v in enumerate(sorted(set(train.user)))}
    item2idx = {v:k for k,v in enumerate(sorted(set(train.item)))}
    uidx2user = {k:v for k,v in enumerate(sorted(set(train.user)))}
    iidx2item = {k:v for k,v in enumerate(sorted(set(train.item)))}
    
    # indexing
    train.user = train.user.map(user2idx)
    train.item = train.item.map(item2idx)
    
    # train 컬럼명 변경
    train.columns=['user_id:token','item_id:token','timestamp:float']
    
    # to_csv
    outpath = f"dataset/train_data"
    os.makedirs(outpath, exist_ok=True)
    train.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False)
    
    
    yamldata=f"""
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, timestamp]

    show_progress : False
    epochs : {args.epochs}
    device : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_args:
        split: {{'RS': [9, 1, 0]}}
        group_by: user
        order: RO
        mode: full
    metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
    topk: {args.top_k}
    valid_metric: Recall@10


    """
    with open("general.yaml", "w") as f:
        f.write(yamldata)
    
    # run
    model_name = args.model_name
    print(f"running {model_name}...")
    start = time.time()
    result = run(args, model_name)
    t = time.time() - start
    print(f"It took {t/60:.2f} mins")
    print(result)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)