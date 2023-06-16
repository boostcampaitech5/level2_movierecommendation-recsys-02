import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color
from recbole.utils.case_study import full_sort_topk
import os
import pdb


def make_config(args, config_path : str):
    general_model = [
        'Pop',
        'ItemKNN',
        'BPR',
        'NeuMF',
        'ConvNCF',
        'DMF',
        'FISM',
        'NAIS',
        'SpectralCF',
        'GCMC',
        'NGCF',
        'LightGCN',
        'DGCF',
        'LINE',
        'MultiVAE',
        'MultiDAE',
        'MacridVAE',
        'CDAE',
        'ENMF',
        'NNCF',
        'RaCT',
        'RecVAE',
        'EASE',
        'SLIMElastic',
        'SGL',
        'ADMMSLIM',
        'NCEPLRec',
        'SimpleX',
        'NCL'
    ]
    sequence_model = [
        'FPMC',
        'GRU4Rec',
        'NARM',
        'STAMP',
        'Caser',
        'NextItNet',
        'TransRec',
        'SASRec',
        'BERT4Rec',
        'SRGNN',
        'GCSAN',
        'GRU4RecF',
        'SASRecF',
        'FDSA',
        'S3Rec',
        'GRU4RecKG',
        'KSR',
        'FOSSIL',
        'SHAN',
        'RepeatNet',
        'HGN',
        'HRM',
        'NPE',
        'LightSANs',
        'SINE',
        'CORE'
    ]
    context_aware_model = [
        'LR',
        'FM',
        'NFM',
        'DeepFM',
        'xDeepFM',
        'AFM',
        'FFM',
        'FwFM',
        'FNN',
        'PNN',
        'DSSM',
        'WideDeep',
        'DIN',
        'DIEN',
        'DCN',
        'DCNV2',
        'AutoInt',
        'XGBOOST',
        'LIGHTGBM'
    ]
    knowledge_based_model = [
        'CKE',
        'CFKG',
        'KTUP',
        'KGAT',
        'KGIN',
        'RippleNet',
        'MCCLK',
        'MKR',
        'KGCN',
        'KGNNLS'
    ]
    

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    model_basic_param_path = os.path.join(curr_dir, 'yaml_dir', 'model_basic_param' + os.path.sep)

    with open(model_basic_param_path + f"{args.model_name}.yaml", 'r') as f:
        model_basic_param = f.read()

    if args.model_name.lower() in list(map(lambda x: x.lower(), general_model)):
        yamldata=f"""field_separator: "\\t"

# dataset config : General Recommendation
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
MAX_ITEM_LIST_LENGTH: {args.max_len}

filter_inter_by_user_or_item: {args.filter_inter}
user_inter_num_interval: "[{args.user_lower_bound},{args.user_upper_bound})"
item_inter_num_interval: "[{args.item_lower_bound},{args.item_upper_bound})"

load_col:
    inter: [user_id, item_id, timestamp]
    user : [user_id]
    item: [item_id, year, writer, title, genre, director]

# model config
{model_basic_param}

# Training and evaluation config
epochs: {args.epochs}
train_batch_size: 4096
eval_batch_size: 4096
train_neg_sample_args:
    distribution: uniform
    sample_num: 1
    alpha: 1.0
    dynamic: False
    candidate_num: 0
eval_args:
    group_by: user
    order: RO
    split: {{'RS': [0.9, 0.1, 0.0]}}
    mode: full

device : torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
topk: 10
valid_metric: Recall@10
metric_decimal_place: 4

log_wandb : True
wandb_project : Recbole
"""

    elif args.model_name.lower() in list(map(lambda x: x.lower(), sequence_model)):
        yamldata=f"""field_separator: "\\t"

# dataset config : Sequential Recommendation
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: {args.max_len}

filter_inter_by_user_or_item: {args.filter_inter}
user_inter_num_interval: "[{args.user_lower_bound},{args.user_upper_bound})"
item_inter_num_interval: "[{args.item_lower_bound},{args.item_upper_bound})"

load_col:
    inter: [user_id, item_id, timestamp]
    user : [user_id]
    item: [item_id, year, writer, title, genre, director]

# model config
{model_basic_param}

# Training and evaluation config
epochs: {args.epochs}
train_batch_size: 4096
eval_batch_size: 4096
train_neg_sample_args: ~
eval_args:
    group_by: user
    order: TO
    split: {{'LS': 'valid_and_test'}}
    mode: full
metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
topk: 10
valid_metric: Recall@10
metric_decimal_place: 4

log_wandb : True
wandb_project : Recbole
"""

    elif args.model_name.lower() in list(map(lambda x: x.lower(), context_aware_model)):
        yamldata=f"""field_separator: "\\t"

# dataset config : Context-aware Recommendation
load_col:
    inter: [user_id, item_id, timestamp]
    user : [user_id]
    item: [item_id, year, writer, title, genre, director]

# model config
{model_basic_param}

# Training and evaluation config
epochs: {args.epochs}
train_batch_size: 4096
eval_batch_size: 4096
eval_args:
  split: {{'RS':[0.9, 0.1, 0.0]}}
  order: RO
  group_by: ~
  mode: labeled
train_neg_sample_args: 
    uniform : 1

device : torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
topk: 10
valid_metric: Recall@10
"""

    elif args.model_name.lower() in list(map(lambda x: x.lower(), knowledge_based_model)):
        yamldata=f"""field_separator: "\\t"

# dataset config : Knowledge-based Recommendation
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id
load_col:
    inter: [user_id, item_id]
    kg: [head_id, relation_id, tail_id]
    link: [item_id, entity_id]

# model config
{model_basic_param}

# Training and evaluation config
eval_args:
   split: {{'RS': [0.9, 0.1, 0.0]}}
   group_by: user
   order: RO
   mode: full

device : torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
topk: 10
valid_metric: Recall@10
"""

    with open(config_path, "w") as f:
        f.write(yamldata)


def load_yaml(args):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(curr_dir, 'yaml_dir', f"{args.model_name}.yaml")

    if not os.path.isfile(yaml_path):
        print("Make config...")
        make_config(args, yaml_path)