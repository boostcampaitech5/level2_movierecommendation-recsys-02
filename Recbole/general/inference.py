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
from util import afterprocessing

from recbole.model.general_recommender.multivae import MultiVAE
from recbole.quick_start import run_recbole

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color



def main(args):
    """모델 inference 파일
    args
        --inference_model(모델경로)로 사용할 모델을 선택합니다.
        --rank_K로 몇개의 추천아이템을 뽑아낼지 선택합니다.
    """
    model_path = args.inference_model
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    config['dataset'] = 'train_data'

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, test_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    # device 설정
    device = config.final_config_dict['device']

    # user, item id -> token 변환 array
    user_id = config['USER_ID_FIELD']
    item_id = config['ITEM_ID_FIELD']
    user_id2token = dataset.field2id_token[user_id]
    item_id2token = dataset.field2id_token[item_id]

    # user id list
    all_user_list = torch.arange(1, len(user_id2token)).view(-1,128)

    # user, item 길이
    user_len = len(user_id2token)
    item_len = len(item_id2token)

    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr')

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None

    # model 평가모드 전환
    model.eval()

    # progress bar 설정
    tbar = tqdm(all_user_list, desc=set_color(f"Inference", 'pink'))

    for data in tbar:
        # interaction 생성
        interaction = dict()
        interaction = Interaction(interaction)
        interaction[user_id] = data
        interaction = interaction.to(device)

        # user item별 score 예측
        score = model.full_sort_predict(interaction)
        score = score.view(-1, item_len)

        rating_pred = score.cpu().data.numpy().copy()

        user_index = data.numpy()

        idx = matrix[user_index].toarray() > 0

        rating_pred[idx] = -np.inf
        rating_pred[:, 0] = -np.inf
        ind = np.argpartition(rating_pred, -args.rank_K)[:, -args.rank_K:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[
            np.arange(len(rating_pred))[:, None], arr_ind_argsort
        ]

        if pred_list is None:
            pred_list = batch_pred_list
            user_list = user_index
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(
                user_list, user_index, axis=0
            )

    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    #데이터 저장
    sub = pd.DataFrame(result, columns=["user", "item"])
     # train load
    train = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
    # indexing save
    uidx2user = {k:v for k,v in enumerate(sorted(set(train.user)))}
    iidx2item = {k:v for k,v in enumerate(sorted(set(train.item)))}
    
    sub.user = sub.user.map(uidx2user)
    sub.item = sub.item.map(iidx2item)
    
    sub = afterprocessing(sub,train)
    # SAVE OUTPUT
    output_dir = os.getcwd()+'/output/'
    write_path = os.path.join(output_dir, f"{args.model_name}.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("user,item\n")
        for id, p in sub.values:
            w.write('{},{}\n'.format(id,p))
    print('inference done!')
    








if __name__ == "__main__":
    args = parse_args()
    main(args)