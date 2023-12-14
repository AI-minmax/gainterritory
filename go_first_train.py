import os
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

proc_num = 24  # 24번까지 존재(컴퓨터 사양에 따라 다름)


def convert_to_bool(value):
    return bool(int(value))


def load_train_data():
    _data = []
    df = pd.read_csv(f"learning_data/data_{0}.csv", index_col=None, header=None)
    for i in range(24):
        _data.append(pd.read_csv(f"learning_data/data_{i}.csv", index_col=None, header=None,
                                 converters={col: convert_to_bool for col in df.columns}))
    _data = pd.concat(_data)
    X, y = _data.iloc[:, :-1], _data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test


def load_train_data2():
    _data = []
    df = pd.read_csv(f"learning_data/data_{0}.csv", index_col=None, header=None)
    for i in range(24):
        _data.append(pd.read_csv(f"learning_data/data_{i}.csv", index_col=None, header=None,
                                 converters={col: convert_to_bool for col in df.columns}))
    _data = pd.concat(_data, ignore_index=True)
    X, y = _data.iloc[:, :-1], _data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, y_train, X_test, y_test


def read_real_board(file_path):
    file_path = "./board_library/sample_5nodes.csv"
    df = pd.read_csv(file_path, index_col=0)
    # df = pd.read_csv(file_path, index_col=0, converters={col: convert_to_bool for col in df.columns})
    return df.to_numpy().flatten().reshape((1,49))


def get_xgbmodel():
    X_train, X_test, y_train, y_test = load_train_data()
    model = XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=40, eval_metric='logloss')
    # model = XGBClassifier()
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # accuracy_score(y_pred, y_test)
    return model



file_path = "./board_library/sample_10nodes.csv"
model = get_xgbmodel()
file_path = "8nodes_map3.csv"
print(model.predict(read_real_board(file_path)))



class PREDICT(nn.Module):

  def __init__(self, config):
    super(PREDICT, self).__init__()
    self.inode = config["input_node"]
    self.hnode = config["hidden_node"]
    self.onode = config["output_node"]
    self.activation = nn.Sigmoid()
    self.linear1 = nn.Linear(self.inode, self.hnode, bias=True)
    self.linear2 = nn.Linear(self.hnode, self.onode, bias=True)

  def forward(self, input_features):
    output1 = self.linear1(input_features)
    hypothesis1 = self.activation(output1)
    output2 = self.linear2(hypothesis1)
    hypothesis2 = self.activation(output2)

    return hypothesis2

def load_dataset():

  train_X, train_y, test_X, test_y = load_train_data2()
  train_X = train_X.to_numpy()
  test_X = test_X.to_numpy()
  train_y= train_y.to_numpy()
  test_y= test_y.to_numpy()

  train_X = torch.tensor(train_X, dtype=torch.float)
  train_y = torch.tensor(train_y, dtype=torch.float)
  test_X = torch.tensor(test_X, dtype=torch.float)
  test_y = torch.tensor(test_y, dtype=torch.float)
  return (train_X, train_y), (test_X, test_y)

def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()


# 평가 수행 함수
def do_test(model, test_dataloader):
  model.eval()
  predicts, golds = [], []
  with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
      batch = tuple(t.cuda() for t in batch)
      input_features, labels = batch
      hyphotesis = model(input_features)
      logits = torch.argmax(hyphotesis,-1)
      x = tensor2list(logits)
      y = tensor2list(labels)
      predicts.extend(x)
      golds.extend(y)
    print("PRED=",predicts)
    print("GOLD=",golds)
    print("Accuracy= {0:f}\n".format(accuracy_score(golds, predicts)))


# 모델 평가 함수
def test(config):
  model = PREDICT(config).cuda()
  # 저장된 모델 가중치 로드
  model.load_state_dict(torch.load(os.path.join(config["output_dir"], config["model_name"])))
  # 데이터 load
  (_, _), (features, labels) = load_dataset()
  test_features = TensorDataset(features, labels)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])
  do_test(model, test_dataloader)


def train(config):
  model = PREDICT(config).cuda()
  (input_features,labels),(_,_)=load_dataset()
  train_features = TensorDataset(input_features, labels)
  train_dataloader = DataLoader(train_features, shuffle=True, batch_size=config["batch_size"])
  loss_func = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=config["learn_rate"])
  for epoch in range(config["epoch"]+1):
    model.train()
    costs = []
    for (step, batch) in enumerate(train_dataloader):

      batch = tuple(t.cuda() for t in batch)
      input_features, labels = batch
      labels.resize(1,2)
      optimizer.zero_grad()
      hyphotesis = model(input_features)
      cost = loss_func(hyphotesis, labels)
      cost.backward()
      optimizer.step()
      costs.append(cost.data.item())
    # 에폭마다 평균 비용 출력하고 모델을 저장
    print("Average Loss= {0:f}".format(np.mean(costs)))
    torch.save(model.state_dict(), os.path.join(config["output_dir"], "epoch_{0:d}.pt".format(epoch)))
    do_test(model, train_dataloader)


if(__name__=="__main__"):
    root_dir = "./"
    output_dir = os.path.join(root_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config = {"mode": "train",
              "model_name":"epoch_{0:d}.pt".format(10),
              "output_dir":output_dir,
              "input_node":49,
              "hidden_node":512,
              "output_node":1,
              "learn_rate":0.001,
              "batch_size":2,
              "epoch":10,
              }
    if(config["mode"] == "train"):
        train(config)
    else:
        test(config)
