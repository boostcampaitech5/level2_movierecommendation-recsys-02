import pandas as pd
import numpy as np
import os
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

def feature_engineering(
    train_data,
    year_data,
    writer_data,
    title_data,
    genre_data,
    director_data
):
    
    train_data = remove_same_title(train_data)
    year_data = year_preprocessing(title_data, year_data)
    title_data = remove_year_for_title(title_data)
    train_data = timestamp_feature(train_data)
    writer_data,director_data,genre_data = merge_list(writer_data,director_data,genre_data)
    year_data = rename_year(year_data)
    # year_data = rename_year(year_data)
    # year_data['pub_year_cat'],year_data['pub_year_nor'] = year_data['pub_year'].copy(),year_data['pub_year'].copy()
    # genre_data = apply_pca_to_genre(genre_data, 2)

    return train_data, year_data, writer_data, title_data, genre_data, director_data

def year_preprocessing(title_data, year_data):
    '''
        title에 있는 연도로 year_data의 결측치를 채움
    '''
    title_data['title_year'] = title_data['title'].str.split(' ').str[-1]
    title_data['title_year'] = title_data['title_year'].str.replace(pat=r"[^0-9]",repl="",regex=True)
    title_data['title_year'] = title_data['title_year'].astype(float)
    
    target_item = []
    target_item = list(year_data[year_data['year'].isna()]['item'])
    
    for target in target_item:
        year_data[year_data['item'] == target]['year'] = title_data[title_data['item'] == target]['title_year']
    
    return year_data

def timestamp_feature(train_data:pd.DataFrame)->pd.DataFrame:
    '''
    timestamp를 날짜 형식에 맞춰서 연산 가능한 형태로 바꿈꿈 
    '''
    train_data['time'] = train_data['time'].apply(lambda x: time.strftime('%Y-%m-%d-%H', time.localtime(int(x))))
    date_df = train_data['time'].str.split("-", expand=True)
    date_df.columns = ['ex_year', 'ex_month', 'ex_day','ex_hour'] 
    train_data = pd.concat([train_data, date_df], axis=1)

    return train_data

def remove_year_for_title(title_data:pd.DataFrame)->pd.DataFrame:
    # title에서 괄호와 괄호 내 문자열 제거
    title_data['title'] = title_data['title'].str.replace(pat = r'\(.*\)|\s-\s.*', repl=r'', regex=True)
    title_data['title'] = title_data['title'].str.replace(pat = r'\, The|\s-\s.*', repl=r'', regex=True)
    title_data['title'] = title_data['title'].str.strip()
    return title_data

def merge_list(writer_data,director_data,genre_data):
    writer_data = writer_data.groupby(by = ['item'])['writer'].apply(list).reset_index(name = 'writer')
    director_data = director_data.groupby(by = ['item'])['director'].apply(list).reset_index(name = 'director')
    genre_data = genre_data.groupby(by = ['item'])['genre'].apply(list).reset_index(name = 'genre')
    return writer_data, director_data, genre_data

def rename_year(year_data:pd.DataFrame)->pd.DataFrame:
    year_data.columns=['item','pub_year']
    return year_data

def apply_pca_to_genre(genre_data, n):
    
    '''
        data: 
            pca 적용할 컬럼이 list 형태로 되어있어야 함
            결측치가 없어야 함
    '''
    
    item = genre_data['item']
    genre = genre_data['genre']
    
    mlb = MultiLabelBinarizer()
    x = pd.DataFrame(mlb.fit_transform(genre), columns=mlb.classes_, index=genre_data.index)
    
    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components).add_prefix('pca_')
    
    final_df = pd.concat([item, principal_df], axis = 1)
    
    return final_df

def remove_same_title(train_data):
    '''
        itemID가 다른데 title이 동일한 item을 더 정보가 많은 ID로 통일
        (64997 -> 34048)
    '''
    train_data.loc[train_data[train_data['item'] == 64997].index, 'item'] = 34048
    
    return train_data

def pub_year_to_normalize(pub_year:int,mean:int=1992.174732,std:int=19.052568):
    pub_year = (pub_year - mean)/std
    return pub_year

def pub_year_to_category(pub_year:int):

    if pub_year <= 1950:
        return 1950
    elif pub_year <= 1960:
        return 1960
    elif pub_year <= 1970:
        return 1970
    elif pub_year <= 1980:
        return 1980
    elif pub_year <= 1990:
        return 1990
    elif pub_year <= 2000:
        return 2000
    elif pub_year <= 2005:
        return 2005
    elif pub_year <= 2010:
        return 2010
    else:
        return 2015