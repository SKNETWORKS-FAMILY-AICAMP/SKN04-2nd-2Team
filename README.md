# SKN04-2nd-2Team
# SK네트웍스 4기 2번째 프로젝트 2조
<hr>
### 팀 소개

### 팀명 | 😎기도가 좋다🙏

### 팀원 소개
<p align="center">
        <img src="https://avatars.githubusercontent.com/말랑곰" width="150" height="150"/>
        <img src="https://avatars.githubusercontent.com/sunblockisneeded" width="150" height="150"/>
        <img src="https://avatars.githubusercontent.com/말랑곰" width="150" height="150"/>
        <img src="https://avatars.githubusercontent.com/말랑곰" width="150" height="150"/>
        
  
<div align="center">
|   &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp;권오셈 &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;오창준  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;박화랑  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |     &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;김효은  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;   | 
        
|&nbsp;  &nbsp;  팀장, DL(MLP,모듈화) &nbsp; &nbsp;| &nbsp;  &nbsp;DL(MLP), README &nbsp;&nbsp; | &nbsp;  &nbsp;EDA, ML(LGBM, XGB) &nbsp;&nbsp;|&nbsp;  &nbsp;  EDA(Z-scale), ML(RF) &nbsp;&nbsp;|


</div>

### 데이터 분석 목표
>
<hr>
- 통신사의 고객별 사용량,요금,고객의 소득, 통화 패턴 등 여러 데이터를 이탈 여부를를 ML,DL모델을 통해 결부시켜, 고객 데이터들 통해 향후 고객 이탈 예측 및 대응방안 강구

## 데이터셋 설명

| 변수명 | 데이터 타입 | 설명 |
|--------|-------------|------|
| CustomerID | int64   | 고객의 고유 식별자 |
| Churn | object | 고객이 서비스를 해지했는지 여부를 나타내는 이진 값 (1: 해지, 0: 유지) |
| MonthlyRevenue | float64 | 월별 고객 매출, 고객이 매달 지불하는 금액 |
| MonthlyMinutes | float64 | 고객이 매달 사용한 통화 시간(분) |
| TotalRecurringCharge | float64 | 월별 반복 청구되는 요금 (기본 요금) |
| DirectorAssistedCalls | float64 | 고객이 콜센터나 고객 지원 서비스를 통해 도움을 받은 통화 횟수 |
| OverageMinutes | float64 | 기본 제공 통화 시간을 초과한 분 수 |
| RoamingCalls | float64 | 로밍 상태에서 걸린 통화 횟수 |
| PercChangeMinutes | float64 | 통화 시간이 전월 대비 몇 퍼센트 변동했는지 |
| PercChangeRevenues | float64 | 수익이 전월 대비 몇 퍼센트 변동했는지 |
| DroppedCalls | float64 | 통화 중 끊긴 횟수 |
| BlockedCalls | float64 | 통화 연결 실패 횟수 |
| UnansweredCalls | float64 | 응답하지 않은 전화 횟수 |
| CustomerCareCalls | float64 | 고객이 고객 서비스에 건 전화 횟수 |
| ThreewayCalls | float64 | 삼자 통화 횟수 |
| ReceivedCalls | float64 | 받은 전화의 횟수 |
| OutboundCalls | float64 | 발신한 전화 횟수 |
| InboundCalls | float64 | 수신한 전화 횟수 |
| PeakCallsInOut | float64 | 피크 시간대 발신 및 수신 전화 횟수 |
| OffPeakCallsInOut | float64 | 비피크 시간대 발신 및 수신 전화 횟수 |
| DroppedBlockedCalls | float64 | 끊긴 통화와 연결 실패 통화의 총합 |
| CallForwardingCalls | float64 | 착신 전환된 통화 횟수 |
| CallWaitingCalls | float64 | 대기 중인 통화 횟수 |
| MonthsInService | int64   | 고객이 서비스에 가입한 기간(월) |
| UniqueSubs | int64 | 고객이 연결된 고유 가입자의 수 |
| ActiveSubs | int64 | 현재 활성 상태인 가입자의 수 |
| ServiceArea | object | 서비스 지역 |
| Handsets | float64 | 사용 중인 휴대폰의 수 |
| HandsetModels | float64 | 사용 중인 휴대폰 모델의 수 |
| CurrentEquipmentDays | float64 | 현재 장비(휴대폰)를 사용한 일수 |
| AgeHH1 | float64 | 가구주 1의 나이 |
| AgeHH2 | float64 | 가구주 2의 나이 |
| ChildrenInHH | float64 | 가구에 있는 자녀 수 |
| HandsetRefurbished | object | 고객이 사용하는 휴대폰이 리퍼폰인지 여부 |
| HandsetWebCapable | object | 고객의 휴대폰이 인터넷 접속이 가능한지 여부 |
| TruckOwner | object | 고객이 트럭을 소유하고 있는지 여부 |
| RVOwner | object | 고객이 RV(레저용 차량)를 소유하고 있는지 여부 |
| Homeownership | object | 고객이 주택을 소유하고 있는지 여부 |
| BuysViaMailOrder | object | 우편 주문을 통해 물건을 구매하는지 여부 |
| RespondsToMailOffers | object | 우편으로 받은 마케팅 제안에 응답하는지 여부 |
| OptOutMailings | object | 우편 수신 거부 여부 |
| NonUSTravel | object | 미국 외의 지역으로 여행을 하는지 여부 |
| OwnsComputer | object | 컴퓨터를 소유하고 있는지 여부 |
| HasCreditCard | object | 신용카드를 소유하고 있는지 여부 |
| RetentionCalls | float64 | 고객 유지팀에 걸려온 전화 횟수 |
| RetentionOffersAccepted | float64 | 고객 유지팀에서 제안한 오퍼를 수락한 횟수 |
| NewCellphoneUser | object | 새 휴대폰 사용자인지 여부 |
| NotNewCellphoneUser | object | 기존 휴대폰 사용자인지 여부 |
| ReferralsMadeBySubscriber | float64 | 고객이 추천한 가입자의 수 |
| IncomeGroup | object | 고객의 소득 수준을 나타내는 그룹 |
| OwnsMotorcycle | object | 고객이 오토바이를 소유하고 있는지 여부 |
| AdjustmentsToCreditRating | float64 | 신용 등급에 대한 조정 사항 |
| HandsetPrice | float64 | 고객이 소유한 휴대폰의 가격 |
| MadeCallToRetentionTeam | object | 고객 유지팀에 전화를 건 적이 있는지 여부 |
| CreditRating | float64 | 고객의 신용 등급 |
| PrizmCode | object | 고객의 사회 경제적 프로파일을 나타내는 코드 |
| Occupation | object | 고객의 직업 |
| MaritalStatus | object | 고객의 결혼 여부 |
| data.shape = (51047, 58)|

--- 


### 데이터 로드 및 결측치| 중복값 확인

```python
# 결측치 확인
data.isna().sum()
```
<img src="./data/isna_sum.png"/>

```python
# 결측치 최대 3515개
# 전부 drop해도 최소 약47000개 정도 남으니까 전부 drop
data.isna().sum().sum() 
data = data.dropna().reset_index(drop=True)
```
```python
data.duplicated().sum() # 중복값 확인| 결과는 0
```
```python
# 본격적으로 들어가기 target데이터 분포 확인
# 불균형함
plt.pie(
    data.Churn.value_counts(),
    explode = [0, 0.05], # 각 조각 사이 거리
    labels = data.Churn.value_counts().index,
    autopct='%1.1f%%', # 퍼센트 표시
    colors=['#dcdcdc', '#87ceeb'],
    shadow=True
)
```
<img src="./data/Churn_pie.png"/>


### EDA 및 전처리

``` python
new_data_1 = data.filter(items=['Churn', 'CustomerID']) # CustomterID가 Index역할을 하고 있는데 혹시 범위별로 묶으면 유의미한 변수일까 확인
new_data_1.CustomerID = new_data_1.CustomerID//40000 # CustomerID 범주화 40000개씩 묶음
new_data_1

sns.countplot(
    new_data_1,
    x= 'CustomerID',
    hue= 'Churn'
)
# 범주를 묶어서 확인해 봤는데 유의미한 변화 X
```
<img src="./data/CustomerID_10.png"/>

```python
# 더 세부적으로 묶음
new_data_1 = data.filter(items=['Churn', 'CustomerID'])
new_data_1.CustomerID = new_data_1.CustomerID//4000 # 이번에는 CustomerID 4000개씩 묶음
sns.countplot(
    new_data_1,
    x= 'CustomerID',
    hue= 'Churn'
)
# 마찬가지로 의미없음 그래서 CustomerID drop
```
<img src="./data/CustomerID_100.png"/>

```python
# 숫자형 데이터 boxplot 찍어보기
# 사분위 수 범위에서 크게 벗어나는 애들 제거
sns.boxplot(
    data.loc[:, (data.dtypes != object)].iloc[:, [16]],
)
```
<img src="./data/Boxplot_before.png"/>

```python
# Z-score 활용해서 이상치 제거
columns_to_zscore = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
       'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls',
       'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls',
       'BlockedCalls', 'UnansweredCalls', 'CustomerCareCalls', 'ThreewayCalls',
       'ReceivedCalls', 'OutboundCalls', 'InboundCalls', 'PeakCallsInOut',
       'OffPeakCallsInOut', 'DroppedBlockedCalls', 'CallForwardingCalls',
       'CallWaitingCalls', 'MonthsInService', 'UniqueSubs', 'ActiveSubs',
       'Handsets', 'HandsetModels', 'CurrentEquipmentDays', 'AgeHH1', 'AgeHH2',
       'RetentionCalls', 'RetentionOffersAccepted',
       'ReferralsMadeBySubscriber', 'IncomeGroup',
       'AdjustmentsToCreditRating']
z_scores = stats.zscore(data[columns_to_zscore])
outliers_mask = (abs(z_scores) > 3).any(axis=1)
data = data[~outliers_mask]

# 만족스럽게 제거 됐는지 확인
sns.boxplot(
    data.loc[:, (data.dtypes != object)].iloc[:, [16]],
)
``` 
<img src="./data/Boxplot_after.png"/>

```python
# 이상치 제거 후 0으로 통일된 값들 확인 후 drop
data.loc[:, (data.sum() == 0)].value_counts()
data = data.drop(columns=['RetentionCalls', 'RetentionOffersAccepted', 'ReferralsMadeBySubscriber'])
```

### 수치형데이터 확률밀도함수 확인 (데이터 분포)
```python
sns.kdeplot(
    data,
    x= data.loc[:, (data.dtypes != object)].columns[0],
    hue='Churn'
)
```
<img src="./data/Kdeplot.png"/>

```python
# 범주형데이터 데이터 분포 확인

sns.countplot(
    data,
    x=data.loc[:, (data.dtypes == object)].columns[15],
    hue='Churn'             
)
```
<img src="./data/Countplot.png"/>

```python
# object형 숫자형으로 바꿔주기
label_encoders = {}
for column in data.loc[:, (data.dtypes == object)].columns:
    label_encoder = LabelEncoder()
    data.loc[:, column] = label_encoder.fit_transform(data[column])

    label_encoders.update({column: label_encoder})

data = data.astype(dict(zip(np.extract(data.dtypes == object , data.columns), ['float']*23)))

# csv 내보내기
data.to_csv('./data/preprocessing_train.csv', index= False)
```
### DEEP LEARNING
>
<hr>

전처리 후 이진 분류에 필요한 모델과 OPTIMIZER, 손실함수 선정
모델 - MLP : 고객행동/특성과 이탈여부 사이의 비선형적인 복잡한 관계를 학습하기 위해서.

옵티마이저 - ADAM : ADAM은 학습률을 자동으로 조정하기에 더 효율적으로 최적화 과정을 진행할 수 있음.
학습 특성이 많은 경우에도, ADAM은 다른 OPTIMIZER보다 빠르게 답에 수렴하면서도 과적합을 방지하는데 도움이 된다고 알려져 있음

손실함수 - binary_cross_entropy(BCE)_with_logits : 이진 분류 문제에 적합한 손실 함수. 내부적으로 Sigmoid와 교차 엔트로피 손실을 함께 계산하기 때문에 그냥 BCE보다 수치적으로 안정적이고 효율적임.



하이퍼파라미터 튜닝 요약
>
<hr>

다음 서술될 유의미한 변화를 관찰할 수 없다는 말은
정확도(Acc), VAL_LOSS, TRAIN_LOSS의 수치상 유의미한 변화를 찾을 수 없음을 의미.


BATCH SIZE

64 ~ 512 유의미한 차이 관찰 X

EPOCHS

PATIENT 5~10 설정시 LOSS 4~6STEP 후 early stop 에 의해 정지
LEARNING RATE

0.001 ~ 0.03 범위에서 유의미한 변화 관찰 X

HIDDEN_DIM

64~512 유의미한 결과 관찰 X
DROPOUT RATIO

0.1에서 0.5까지 유의미한 경향 X
은닉층 개수 2개 ~7개

유의미한 변화 관찰 X
NNI를 통한 최종 HYPERPARAM 조합
<div align="center">
<img src=".\data/NNI_hyperparam_result.png">
</div>


HIDDEN LAYER 개수 4, EPOCHS 10, EARLY-STOP 작동X
<div align="center">
        
실행결과
<img src="./data/acc_result.png">

cross_validation 결과
<img src="./data/matrix_result.png">
</div>


Deel Learning결과
test data의 0 대 1 비율은 약 71: 29 인데 acc가 0.71
결과를 보면 0의 재현율이 recall이 1
loss값은 줄어드는 모습을 보임
1,2,3 을 종합해 보았을 때, 학습이 만족스러운 수준으로 일어나지 않았고, 많은 값들이 1으로 올라가지 못하는 모습을 관찰할 수 있음.

### ML 사용기법: LGBMClassifier, XGBClassifier, RandomForestClassifier

선정이유: Tree계열 모델을 활용하여 Feature_importace 뽑아 어떤 변수가 고객이탈에 유의미한 영향을 끼치는 지 확인

```python

for i, (train_index, test_index) in enumerate(stratified_k_fold.split(data, data.Churn)):
    temp = data.iloc[train_index]
    test = data.iloc[test_index]
    train, valid = train_test_split(temp, test_size= len(test)/len(temp), random_state=0)

    rf = RandomForestClassifier(random_state=0)
    
    parameters = {
        'max_depth': np.random.randint(1, 15, 5),
        'n_estimators': np.random.randint(50, 500, 20),
        }

    random_search = RandomizedSearchCV(rf, parameters, random_state=0)
    random_search.fit(train.drop(columns=['Churn']), train.Churn)

    index_best_param = random_search.cv_results_.get('rank_test_score').argmax()
    best_params = random_search.cv_results_.get('params')[index_best_param]

    rf =RandomForestClassifier(
        n_estimators = best_params.get('n_estimators'),
        max_depth = best_params.get('max_depth'),
        random_state=0
    ).fit(train.drop(columns=['Churn']), train.Churn)

    confusion_matrix_result = confusion_matrix(
        test.Churn,
        rf.predict(test.drop(columns=['Churn']))
    )

```
# 모델 학습 결과
### LGBMClassifier
| Fold | precision | recall | f1 | accuracy |best_importances|
|------|-----------|--------|----|----------|----------------|
|fold1| 0.71 | 0.99 |0.83| 0.71| MonthsInService| 
|fold2| 0.74 | 0.91 |0.82| 0.71| PercChangeMinutes|
|fold3| 0.74 | 0.92 |0.82| 0.71| PercChangeMinutes|
|fold4| 0.71 | 1.0  |0.83| 0.71| MonthsInService| 
|fold5| 0.74 | 0.93 |0.82| 0.72| PercChangeMinutes|

### XGBClassifier
| Fold | precision | recall | f1 | accuracy |best_importances|
|------|-----------|--------|----|----------|----------------|
|fold1| 0.74 | 0.90 |0.81| 0.70| HandsetRefurbished| 
|fold2| 0.75 | 0.89 |0.81| 0.70| HandsetRefurbished|
|fold3| 0.74 | 0.89 |0.81| 0.70| HandsetRefurbished|
|fold4| 0.74 | 0.90 |0.81| 0.70| HandsetWebCapable | 
|fold5| 0.74 | 0.90 |0.81| 0.70| HandsetRefurbished|

### RandomForestClassifier
| Fold | precision | recall | f1 | accuracy |best_importances|
|------|-----------|--------|----|----------|----------------|
|fold1| 0.72 | 1.0  |0.83| 0.72| MonthsInService| 
|fold2| 0.72 | 1.0  |0.83| 0.72| PercChangeMinutes|
|fold3| 0.72 | 1.0  |0.83| 0.72| PercChangeMinutes|
|fold4| 0.72 | 1.0  |0.84| 0.72| MonthsInService| 
|fold5| 0.72 | 1.0  |0.83| 0.72| PercChangeMinutes|


# 결론

### 딥러닝과 머신러닝을 비교해 보았을 때 acc는 머신러닝이 더 높게 나옴(RandomForestClassifier)
### 하지만 0.71(MLP)과 0.72(RandomForestClassifier)차이 이므로 그렇게 유의미한 차이는 아님
### acc: 0.8을 넘지 못함
### 유의미한 변수를 찾지 못함
### 유의미한 칼럼이 빠졌거나 애초에 Churn을 잘못수집했을 가능성이 있음
