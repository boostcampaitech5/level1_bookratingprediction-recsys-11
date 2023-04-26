import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
from src.data_.cat_data import cat_data_load, cat_data_split
from src.train import train, test
import numpy as np
import warnings
import re
from catboost import CatBoostRegressor

def modify_range(rating):
  if rating < 0:
    return 0
  elif rating > 10:
    return 10
  else:
    return rating

def rmse(real, predict):
  pred = list(map(modify_range, predict))  
  pred = np.array(pred)
  return np.sqrt(np.mean((real-pred) ** 2))


def main(args):
    Setting.seed_everything(args.seed)
    
    params = {         
            'learning_rate': args.lr,
            'iterations' : args.iterations,
            'depth' : args.depth,
            'l2_leaf_reg' : args.l2_leaf_reg,
            'task_type' :  'GPU',
            'devices' : '0:1'
            }

    #'lambda': args.lambda_,

    ######################## DATA LOAD
    print(f'--------------- CatBoost Load Data ---------------')
    data = cat_data_load(args)

    
    ######################## Train/Valid Split
    print(f'--------------- CatBoost Train/Valid Split ---------------')
    data = cat_data_split(args, data)
     #print(len(data['X_train']),len(data['rating_train']))
    
    

    ######################## TRAIN
    print(f'--------------- CatBoost TRAINING ---------------')
    catboost_r = CatBoostRegressor(**params, verbose=True, random_state=42)
    # print(args.depth)
    catboost_r.fit(data['X_train'].select_dtypes(exclude='object'), data['y_train'], eval_set = [(data['X_valid'].select_dtypes(exclude='object'), data['y_valid'])],early_stopping_rounds=args.esr, verbose=100, use_best_model=True)
    # return
    

    ######################## INFERENCE
    print(f'--------------- CatBoost PREDICT ---------------')
    predicts = catboost_r.predict(data['X_test'].select_dtypes(exclude='object'))


    ######################## SAVE PREDICT
    print(f'--------------- SAVE CatBoost PREDICT ---------------')
    # data['rating_test']['user_id']= data['rating_test']['user_id'].map(data['idx2user'])
    # print(data['rating_test'].head())
    
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    # print(submission.head())
    submission['rating'] = predicts


    setting = Setting()
    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, default='CatBoost',
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')


    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=18, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=0.03, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')
    arg('--l2_leaf_reg',type=float, default=3, help='catboost params')
    arg('--iterations',type=float, default=1000, help='catboost params')
    arg('--depth',type=float, default=6, help='catboost params')
    arg('--esr',type=int, default=1000, help='catboost.fit params')
    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    

    args = parser.parse_args()
    # print(args)
    main(args)
