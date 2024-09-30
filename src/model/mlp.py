import torch.nn as nn
    
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs.get('input_dim')
        self.hidden_dim = configs.get('hidden_dim')
        self.output_dim = configs.get('output_dim')
        self.dropout_ratio = configs.get('dropout_ratio')
        self.use_batch_norm = configs.get('use_batch_norm')

        # 레이어 1
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_dim) if self.use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=self.dropout_ratio)

        # 레이어 2
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim )
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_dim) if self.use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout_ratio)

        # 레이어 3
        self.linear3 = nn.Linear(self.hidden_dim  , self.hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(self.hidden_dim) if self.use_batch_norm else nn.Identity()
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=self.dropout_ratio)

        

        # 출력 레이어
        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        # 이진 분류를 위한 Sigmoid 활성화는 손실 함수에서 처리합니다.

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.output(x)
        return x
