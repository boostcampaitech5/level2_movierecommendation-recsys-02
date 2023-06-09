import os, pdb, pickle
import pandas as pd

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

    # indexing save
    user2uidx, item2iidx, _, _ = load_index_file()
    
    # indexing
    train_data.user = train_data.user.map(user2uidx)
    train_data.item = train_data.item.map(item2iidx)

    df_merge = pd.merge(train_data, title_data, on='item', how='left')
    df_merge = pd.merge(df_merge, year_data, on='item', how='left')
    df_merge = pd.merge(df_merge, writer_data, on='item', how='left')
    df_merge = pd.merge(df_merge, genre_data, on='item', how='left')
    df_merge = pd.merge(df_merge, director_data, on='item', how='left')

    user_data = df_merge[['user']].drop_duplicates(subset=['user']).reset_index(drop=True)
    item_data = df_merge[['item', 'title', 'year', 'writer', 'genre', 'director']].drop_duplicates(subset=['item']).reset_index(drop=True)
    
    return train_data, user_data, item_data

def save_atomic_file(train_data, user_data, item_data):
    dataset_name = 'train_data'
    # train_data 컬럼명 변경
    train_data.columns = ['user_id:token','item_id:token','timestamp:float']
    user_data.columns = ['user_id:token']
    item_data.columns = ['item_id:token', 'title:token_seq', 'year:token', 'writer:token', 'genre:token', 'director:token']
    
    # to_csv
    outpath = f"dataset/{dataset_name}"
    os.makedirs(outpath, exist_ok=True)
    train_data.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False)
    train_data.to_csv(os.path.join(outpath,"train_data.item"),sep='\t',index=False)
