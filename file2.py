import matplotlib as pl
import numpy as np
from apyori import apriori
import pandas as pd
import xgboost as xgb

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('train.csv')



params = {"objective":"multi:softprob",
          "num_class":38,
          "eta":0.1,
          "max_depth":12,
          "min_child_weight":3}
gbm = xgb.train(params,train_data,num_boost_round = 300,early_stopping_rounds = 10,verbose_eval=True)

