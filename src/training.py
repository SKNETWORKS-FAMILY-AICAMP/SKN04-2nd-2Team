import numpy as np
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

class ChurnModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,          # 모델 객체 (nn.Module을 상속받은 모델)
        configs: dict,
    ):
        super().__init__()
        self.model = model         # 모델을 초기화
        self.configs = configs
        self.learning_rate = configs.get('learning_rate')  # 학습률을 초기화

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.test_losses = []
        self.test_accs = []

    def training_step(self, batch, batch_idx):
       
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y')  # 레이블 데이터를 가져옴
        y = y.squeeze()

        #모델을 통해 예측값 계산
        output = self.model(X).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, y)
        
        #예측값 추가
        predicted = torch.sigmoid(output) > 0.5

        self.acc = (predicted == y).float().mean()

        self.train_losses.append(loss.item())
        self.train_accs.append(self.acc.item())
        
        return {"loss": loss, "acc": self.acc}
    
    def on_train_epoch_end(self, *args, **kwargs):

        avg_loss = sum(self.train_losses) / len(self.train_losses)
        avg_acc = sum(self.train_accs) / len(self.train_accs)
        
        self.log_dict(
            {'loss/train_loss': avg_loss, 'acc/train_acc': avg_acc},
            on_epoch=True,
            prog_bar=True,  
            logger=True,    
        )

        self.train_losses.clear()
        self.train_accs.clear()
    
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.val_losses.clear()
            self.val_accs.clear()
        # 검증 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y').squeeze()  # 레이블 데이터를 가져옴
    
        #계산
        output = self.model(X).squeeze()

        self.val_loss = F.binary_cross_entropy_with_logits(output, y)  
        self.val_losses.append(self.val_loss.item())

        predicted = torch.sigmoid(output) > 0.5
        self.val_acc = (predicted == y).float().mean()
        self.val_accs.append(self.val_acc.item())

        return {"val_loss": self.val_loss, "val_acc": self.val_acc}  
    
    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.val_losses) / len(self.val_losses)
        avg_val_acc = sum(self.val_accs) / len(self.val_accs)
        
        self.log_dict(
        {'loss/val_loss': avg_val_loss,  # 평균 검증 손실
         'acc/val_acc': avg_val_acc,    # 평균 검증 정확도
         'learning_rate': self.learning_rate},  # 학습률도 로그에 기록
        on_epoch=True,
        prog_bar=True,  # 진행 막대에 표시
        logger=True,    # 로그에 기록
    )
        
        if self.configs.get('nni'):
            nni.report_intermediate_result(np.mean(self.val_losses))
        
        self.val_losses.clear()
        self.val_accs.clear()


    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_losses.clear()
            self.test_accs.clear()

        
        X = batch.get('X')  # 입력 데이터를 가져옴
        y = batch.get('y').squeeze()  # 레이블 데이터를 가져옴
        

        output = self.model(X).squeeze()  # 모델을 통해 예측값을 계산

        test_loss = F.binary_cross_entropy_with_logits(output, y)
        self.test_losses.append(test_loss.item())
        

        predicted = torch.sigmoid(output) > 0.5
        self.test_acc = (predicted == y).float().mean()
        self.test_accs.append(self.test_acc.item())


        return {"test_loss": test_loss, "test_acc": self.test_acc}
    

    def on_test_epoch_end(self):
        test_loss_average = np.mean(self.test_losses)
        test_accuracy_average = np.mean(self.test_accs)


        self.log('test_loss', test_loss_average)
        self.log('test_accuracy', test_accuracy_average)

        print(f'total test loss: {test_loss_average}')
        print(f'total test acc: {test_accuracy_average}')

        if self.configs.get('nni'):
            nni.report_final_result(test_loss_average)
        self.test_losses.clear()
        self.test_accs.clear()

    def configure_optimizers(self):
        # 옵티마이저와 스케줄러를 설정하는 메서드
        optimizer = optim.Adam(
            self.model.parameters(),  # 모델의 파라미터를 옵티마이저에 전달
            lr=self.learning_rate,    # 학습률 설정
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',               # 손실이 감소할 때 학습률을 줄임
            factor=0.5,               # 학습률 감소 비율
            patience=3,               # 손실이 감소하지 않을 때 대기 에포크 수
        )

        return {
            'optimizer': optimizer,   # 옵티마이저 반환
            'lr_scheduler' : {
            'scheduler': scheduler,
            'monitor': 'loss/val_loss',
            'interval' : 'epoch',
            'frequency' : 1,
            }
        }
    
    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch['X']
        return self(x)
