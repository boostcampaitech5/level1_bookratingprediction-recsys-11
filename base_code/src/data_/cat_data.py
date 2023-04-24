import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_cat_data(users, books, ratings1, ratings2):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'summary_topic']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'summary_topic']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'summary_topic']], on='isbn', how='left')

    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    summary_topic2idx = {v:k for k,v in enumerate(context_df['summary_topic'].unique())}
    
    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['summary_topic'] = train_df['summary_topic'].map(summary_topic2idx)
    
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['summary_topic'] = test_df['summary_topic'].map(summary_topic2idx)
     


    idx = {
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "summary_topic" : summary_topic2idx 
            }

    return idx, train_df, test_df


def cat_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    # books.fillna(-1,inplace=True)
    # users.fillna(-1,inplace=True)


    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_cat_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6,  len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['summary_topic'])], dtype=np.uint32)
    # print(context_test.columns)
    data = {
            'train':context_train,
            'test':context_test,
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def cat_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """
    # print(data['test'].columns)
    rating_test = data['test'][['user_id', 'isbn','rating']]
    X_test = data['test'].drop(['user_id', 'isbn','rating'], axis = 1)
    
    X_train, X_valid, y_train, y_valid,  = train_test_split(
                                                        data['train'],
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    # print(X_valid.columns)
    
    
    rating_train = X_train[['user_id', 'isbn', 'rating']]
    rating_valid = X_valid[['user_id', 'isbn', 'rating']]
    
    X_train = X_train.drop(['user_id', 'isbn', 'rating'], axis = 1)
    X_valid = X_valid.drop(['user_id', 'isbn', 'rating'], axis = 1)


    
    data['X_test'] = X_test
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    data['rating_train'], data['rating_test'], data['rating_valid'] = rating_train, rating_test, rating_valid
    return data




