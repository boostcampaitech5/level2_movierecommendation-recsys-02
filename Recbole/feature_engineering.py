import pandas as pd
import numpy as np
import os

def feature_engineering(
    train_data,
    year_data,
    writer_data,
    title_data,
    genre_data,
    director_data
):
    
    year_data = year_preprocessing(title_data, year_data)
    title_data = remove_year_for_title(title_data)
    train_data = timestamp_feature(train_data)
    writer_data,director_data,genre_data = merge_list(writer_data,director_data,genre_data)
    
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
    writer_data = writer_data.groupby(by = ['item'])['writer'].apply(list).reset_index(name = 'genre')
    director_data = director_data.groupby(by = ['item'])['director'].apply(list).reset_index(name = 'genre')
    genre_data = genre_data.groupby(by = ['item'])['genre'].apply(list).reset_index(name = 'genre')
    return writer_data,director_data,genre_data