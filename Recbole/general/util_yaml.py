import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color
from recbole.utils.case_study import full_sort_topk
import os



def make_config(args, config_name : str) -> None:

    yamldata="""
    field_separator: "\t"
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp

    ITEM_LIST_LENGTH_FIELD: item_length
    LIST_SUFFIX: _list
    MAX_ITEM_LIST_LENGTH: {args.max_len}

    filter_inter_by_user_or_item = {args.filter_inter}
    user_inter_num_interval = [{args.user_lower_bound},{args.user_upper_bound})
    item_inter_num_interval = [{args.item_lower_bound},{args.item_upper_bound})

    load_col:
        inter: [user_id, item_id, timestamp]
        user : [user_id]
        item: [item_id, year, writer, title, genre, director]

    train_neg_sample_args:
        distribution : uniform
        sample_num : 1
    
    show_progress : True
    device : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
    topk: 10
    valid_metric: Recall@10
    
    stopping_step : 10
    
    log_wandb : True
    wandb_project : Recbole
    """

    parameter_dict = args.__dict__

    

    with open(f"{config_name}", "w") as f:
        f.write(yamldata)

    seq_list = set('FPMC', 'GRU4Rec', 'NARM', 'STAMP', 
                'Caser', 'NextItNet', 'TransRec', 'SASRec',
                 'BERT4Rec', 'SRGNN', 'GCSAN','GRU4RecF', 
                 'SASRecF', 'FDSA', 'S3Rec')
    
    if args.loss_type == 'ce':
        parameter_dict['loss_type'] = "CE"
    elif args.loss_type == 'bpr':
        parameter_dict['loss_type'] = "BPR"
    elif args.loss_type == 'bce':
        parameter_dict['loss_type'] = 'BCE'

    if args.model_name not in seq_list:
        parameter_dict['eval_args'] = {
            'split': {'RS': [args.train, 10 -args.train, 0]},
            'group_by': 'user',
            'order': 'RO',
            'mode': 'full',}
    
    # Sequential 모델일 경우 eval_args와 loss_type을 변경
    if args.model_name in seq_list:
        parameter_dict['eval_args'] = {
            'split': {'RS': [args.train, 10 -args.train, 0]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full',}
    
    # inference가 필요한 모델일 경우 1:0:0 학습 변경
    if args.infer:
        parameter_dict['eval_args']['split'] = {'RS' : [1,0,0]}
    
    return parameter_dict
