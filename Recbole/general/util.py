import os, pdb, pickle
import pandas as pd
from datetime import datetime

def make_config(config_name : str) -> None:
    yamldata="""
    field_separator: "\t"
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp
    
    load_col:
        inter: [user_id, item_id, timestamp]
        user : [user_id]
        item: [item_id, year, writer, title, genre, director]

    train_neg_sample_args:
        distribution : uniform
        sample_num : 1

    selected_features: [year, writer, title, genre, director]
    
    show_progress : False
    device : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP']
    topk: 10
    valid_metric: Recall@10
    
    stopping_step : 10
    
    """
    
    with open(f"{config_name}", "w") as f:
        f.write(yamldata)

    return
def load_index_file():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    train_data_path = os.path.join('/opt/ml/input/data/train', 'train_ratings.csv')
    user2uidx_path = os.path.join(curr_path, 'index', 'user2uidx.pickle')
    item2iidx_path = os.path.join(curr_path, 'index', 'item2iidx.pickle')
    uidx2user_path = os.path.join(curr_path, 'index', 'uidx2user.pickle')
    iidx2item_path = os.path.join(curr_path, 'index', 'iidx2item.pickle')
    
    os.makedirs(os.path.join(curr_path, 'index'), exist_ok=True)
    
    if not os.path.isfile(user2uidx_path):
        if not 'train_data' in locals():
            train_data = pd.read_csv(train_data_path)
        user2uidx = {v:k for k,v in enumerate(sorted(set(train_data.user)))}
        with open(user2uidx_path,'wb') as fw:
            pickle.dump(user2uidx, fw)
        fw.close()
    else:
        with open(user2uidx_path, 'rb') as fr:
            user2uidx = pickle.load(fr)
        fr.close()

    if not os.path.isfile(item2iidx_path):
        if not 'train_data' in locals():
            train_data = pd.read_csv(train_data_path)
        item2iidx = {v:k for k,v in enumerate(sorted(set(train_data.item)))}
        with open(item2iidx_path,'wb') as fw:
            pickle.dump(item2iidx, fw)
        fw.close()
    else:
        with open(item2iidx_path, 'rb') as fr:
            item2iidx = pickle.load(fr)
        fr.close()

    if not os.path.isfile(uidx2user_path):
        if not 'train_data' in locals():
            train_data = pd.read_csv(train_data_path)
        uidx2user = {k:v for k,v in enumerate(sorted(set(train_data.user)))}
        with open(uidx2user_path,'wb') as fw:
            pickle.dump(uidx2user, fw)
        fw.close()
    else:
        with open(uidx2user_path, 'rb') as fr:
            uidx2user = pickle.load(fr)
        fr.close()

    if not os.path.isfile(iidx2item_path):
        if not 'train_data' in locals():
            train_data = pd.read_csv(train_data_path)
        iidx2item = {k:v for k,v in enumerate(sorted(set(train_data.item)))}
        with open(iidx2item_path,'wb') as fw:
            pickle.dump(iidx2item, fw)
        fw.close()
    else:
        with open(iidx2item_path, 'rb') as fr:
            iidx2item = pickle.load(fr)
        fr.close()

    return user2uidx, item2iidx, uidx2user, iidx2item

def load_data_file():
    data_path = '/opt/ml/input/data/train'
    # train load
    train_data = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
    title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
    year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    writer_data = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    director_data = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')
    writer_data_group = writer_data.groupby('item', as_index=False).agg(lambda x: ' '.join(set(x)))
    genre_data_group = genre_data.groupby('item', as_index=False).agg(lambda x: ' '.join(set(x)))
    director_data_group = director_data.groupby('item', as_index=False).agg(lambda x: ' '.join(set(x)))
    
    # indexing save
    user2uidx, item2iidx, _, _ = load_index_file()
    
    # indexing
    train_data.user = train_data.user.map(user2uidx)
    train_data.item = train_data.item.map(item2iidx)

    df_merge = pd.merge(train_data, title_data, on='item', how='left')
    df_merge = pd.merge(df_merge, year_data, on='item', how='left')
    df_merge = pd.merge(df_merge, writer_data_group, on='item', how='left')
    df_merge = pd.merge(df_merge, genre_data_group, on='item', how='left')
    df_merge = pd.merge(df_merge, director_data_group, on='item', how='left')

    user_data = df_merge[['user']].drop_duplicates(subset=['user']).reset_index(drop=True)
    item_data = df_merge[['item', 'title', 'year', 'writer', 'genre', 'director']].drop_duplicates(subset=['item']).reset_index(drop=True)
    
    return train_data, user_data, item_data

def save_atomic_file(train_data, user_data, item_data):
    dataset_name = 'train_data'
    # train_data 컬럼명 변경
    train_data.columns = ['user_id:token','item_id:token','timestamp:float']
    user_data.columns = ['user_id:token']
    item_data.columns = ['item_id:token', 'title:token_seq', 'year:token', 'writer:token_seq', 'genre:token_seq', 'director:token_seq']
    
    # to_csv
    outpath = f"dataset/{dataset_name}"
    os.makedirs(outpath, exist_ok=True)
    train_data.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False)
    item_data.to_csv(os.path.join(outpath,"train_data.item"),sep='\t',index=False)
    user_data.to_csv(os.path.join(outpath,"train_data.user"),sep='\t',index=False)


def afterprocessing(sub,train):
    # 날짜를 datetime 형식으로 변환
    new_train = train.copy()
    new_train['time'] = new_train['time'].apply(lambda x: datetime.fromtimestamp(x))
    
    # 이전 시청 영화 제거
    sub = pd.merge(sub,train,on =['user','item'],how='left')
    sub = sub[sub['time'].isnull()][['user','item']]
    # 유저별 영화시청 마지막년도 추출
    user_mv_idx= new_train.groupby('user')['time'].max().reset_index()
    user_mv_idx['lastyear'] = user_mv_idx['time'].apply(lambda x : x.year)
    user_mv_idx.drop('time',inplace = True ,axis=1)

    # 영화 개봉년도와 유저시청년도 합친 데이터프레임 구축
    years = pd.read_csv("/opt/ml/input/data/train/years.tsv",sep = '\t')
    sub = pd.merge(sub,years, on = ['item'] , how = 'left')
    sub = pd.merge(sub,user_mv_idx,on =['user'],how ='left')

    # 늦게 개봉한 영화 제외하고 상위 10개 추출
    sub = sub[sub['lastyear'] >= sub['year']]
    sub = sub.groupby('user').head(10)[['user','item']]
    return sub
