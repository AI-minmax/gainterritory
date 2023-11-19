import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
header = []
for i in range (7):
    for j in range(7):
        header.append(str(i)+str(j))
data = pd.read_csv("DATA.csv", header=["00","01"])
# 전처리 필요함

num_data = len(data)
X = data[:, :num_data - 1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)
model = XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=4, random_state = 32)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred, y_test)
########################
#베이지안 방식으로 정확도를 개선시킬필요 있음

bayes_dtrain = xgb.DMatrix(X_train, y_train)
bayes_dtest = xgb.DMatrix(X_test, y_test)

onehot_encoder=OneHotEncoder()
encoded_cat_matrix=onehot_encoder.fit_transform(X)

param_bounds = {'max_depth': (4, 8),
                'subsample': (0.6, 0.9),
                'colsample_bytree': (0.7, 1.0),
                'min_child_weight': (5, 7),
                'gamma': (8, 11),
                'reg_alpha': (7, 9),
                'reg_lambda': (1.1, 1.5),
                'scale_pos_weight': (1.4, 1.6)}

fixed_params = {'objective': 'binary:logistic',
                'learning_rate': 0.2,
                'random_state': 1991}


#평가지표
def evalfunc(y_true, y_pred):
    pass


def eval_function(max_depth, subsample, colsample_bytree, min_child_weight,
                  reg_alpha, gamma, reg_lambda, scale_pos_weight):
    params = {'max_depth': int(round(max_depth)),
              'subsample': subsample,
              'colsample_bytree': colsample_bytree,
              'min_child_weight': min_child_weight,
              'gamma': gamma,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'scale_pos_weight': scale_pos_weight}

    # 값이 고정된 하이퍼파라미터도 추가
    params.update(fixed_params)

    # XGBoost 모델 훈련 ⓑ
    xgb_model = xgb.train(params=params,
                          dtrain=bayes_dtrain,
                          num_boost_round=2000,
                          evals=[(bayes_dvalid, 'bayes_dvalid')],  # ⓒ
                          maximize=True,  # ⓓ
                          feval=gini,
                          early_stopping_rounds=200,
                          verbose_eval=False)

    best_iter = xgb_model.best_iteration  # 최적 반복 횟수 ⓔ
    # 검증 데이터로 예측 수행 ⓕ
    preds = xgb_model.predict(bayes_dvalid,  # ⓖ
                              iteration_range=(0, best_iter))  # ⓗ

    # 계산
    estimate_score = evalfunc(y_valid, preds)
    print(f'평가점수 : {estimate_score}\n')

    return estimate_score


optimizer = BayesianOptimization(f=eval_function,
                                 pbounds=param_bounds,
                                 random_state=0)

# 베이지안 최적화 수행
optimizer.maximize(init_points=3, n_iter=6)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1991)
oof_val_preds = np.zeros(X.shape[0])
oof_test_preds = np.zeros(X_test.shape[0])