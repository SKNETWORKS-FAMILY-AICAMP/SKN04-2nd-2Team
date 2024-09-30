from src.data import ChurnDataset, ChurnDataModule
from src.utils import convert_category_into_integer
from src.model.mlp import Model
from src.training import ChurnModule

import pandas as pd
import numpy as np
import random
import json
import nni
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

import seaborn as sns



    

def main(configs):
    data = pd.read_csv('./data/preprocessing_train.csv')

    col = list(data.columns)
    for column in ('Churn','ServiceArea', 'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner', 'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 'NotNewCellphoneUser', 'OwnsMotorcycle', 'HandsetPrice', 'MadeCallToRetentionTeam', 'CreditRating', 'PrizmCode','Occupation', 'MaritalStatus'):
        col.remove(column)
# 데이터프레임을 float32로 변환
    data = data.astype(np.float32)

    # 데이터셋을 학습용과 임시 데이터로 분할
    train, temp = train_test_split(data, test_size=0.4, random_state=seed)

    # 임시 데이터를 검증용과 테스트용 데이터로 분할
    valid, test = train_test_split(temp, test_size=0.5, random_state=seed)

    standard_scaler = StandardScaler()

    train.loc[:, col] = standard_scaler.fit_transform(train.loc[:, col])
    valid.loc[:, col] = standard_scaler.transform(valid.loc[:, col])
    test.loc[:, col] = standard_scaler.transform(test.loc[:, col])

    # 데이터셋 객체로 변환
    train_dataset = ChurnDataset(train)
    valid_dataset = ChurnDataset(valid)
    test_dataset = ChurnDataset(test)

    # 데이터 모듈 생성 및 데이터 준비
    Churn_data_module = ChurnDataModule(batch_size=configs.get('batch_size'))
    Churn_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 생성
    configs.update({'input_dim': len(data.columns)-1})
    model = Model(configs)

    # LightningModule 인스턴스 생성
    Churn_module = ChurnModule(
        model=model,
        configs=configs,
    )

    # Trainer 인스턴스 생성 및 설정
    del configs['output_dim'], configs['seed']
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=3)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'Churn/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=Churn_module,
        datamodule=Churn_data_module,
    )
    trainer.test(
        model=Churn_module,
        datamodule=Churn_data_module,
    )

    Churn_module.eval()

    predictions = trainer.predict(
        model=Churn_module,
        dataloaders=Churn_data_module.test_dataloader(),
    )

    actuals = []
    for batch in Churn_data_module.test_dataloader():
        y = batch['y']
        actuals.append(y)

    predictions = torch.cat(predictions)
    actuals = torch.cat(actuals)

    predictions = predictions.cpu().numpy()
    actuals = actuals.cpu().numpy()

    if predictions.ndim > 1:
        predictions = (predictions > 0.5).astype(int)
    precision = precision_score(actuals, predictions, average='weighted')
    recall = recall_score(actuals, predictions, average='weighted')
    f1 = f1_score(actuals, predictions, average='weighted')

    print(f"precision: {precision: .2f} \nrecall: {recall: .2f} \nf1: {f1: .2f} \naccuracy: {(actuals == predictions).mean(): .2f}")

    print(
        classification_report(
            actuals,
            predictions,
            digits=4
        )
    )


if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda', 그렇지 않으면 'cpu' 사용
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    if configs.get('nni'):
        nni_params = nni.get_next_parameter()
        configs.update(nni_params)

    # seed 설정
    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA 설정
    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(configs)