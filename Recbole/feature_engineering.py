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

