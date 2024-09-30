# SK네트웍스 4기 2번째 프로젝트 2조
<hr>
### 팀 소개

### 팀명 : 😎기도가 좋다🙏

### 팀원 소개
<p align="center">
        <img src="https://avatars.githubusercontent.com/말랑곰" width="150" height="150"/>
        <img src="https://avatars.githubusercontent.com/sunblockisneeded" width="150" height="150"/>
        <img src="https://avatars.githubusercontent.com/말랑곰" width="150" height="150"/>
        <img src="https://avatars.githubusercontent.com/말랑곰" width="150" height="150"/>
        
  
<div align="center">
|   &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  &nbsp;권오셈 &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;오창준  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |      &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;박화랑  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;    |     &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;김효은  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;  &nbsp;  &nbsp;   | 
|:------------------------------------------:|:--------------------------------------:|:------------------------------------------:|:-----------------------------------:|
|DB 및 총괄|크롤링 / streamlit|크롤링 / streamlit|크롤링 / streamlit|
</div>

### 데이터 분석 목표
>
<hr>
- 통신사의 고객별 사용량,요금,고객의 소득, 통화 패턴 등 여러 데이터를 이탈 여부와 ML,DL모델을 통해 결부시켜, 고객 데이터들 통해 향후 고객 이탈 예측 및 대응방안 강구


### EDA

-EDA 여기에

### DEEP LEARNING
>
<hr>

- 전처리 후 이진 분류에 필요한 모델과 OPTIMIZER, 손실함수 선정
- 모델 : MLP
- 옵티마이저 : ADAM
- 손실함수 : binary_cross_entropy_with_logits 사용 
### 하이퍼파라미터 튜닝 요약
>
<hr>

다음 서술될 유의미한 변화를 관찰할 수 없다는 말은
정확도, VAL_LOSS, TRAIN_LOSS에서 유의미한 변화를 찾을 수 없음을 의미.
- BATCH SIZE
> 
64 ~ 512 유의미한 차이 관찰 X

- EPOCHS
> 
PATIENT 5~10 설정시 LOSS 4~6STEP 후 early stop 에 의해 정지
- LEARNING RATE
> 
0.001 ~ 0.03 범위에서 유의미한 변화 관찰 X

- HIDDEN_DIM
> 
64~512 유의미한 결과 관찰 X
- DROPOUT RATIO 
> 
0.1에서 0.5까지 유의미한 경향 X
- 은닉층 개수 2개 ~7개 
> 
유의미한 변화 관찰 X
NNI를 통한 최종 HYPERPARAM 조합
<img src=".\NNI_hyperparam_result.png" width="350" height="350">
-

>













### ML

-ML여기에

