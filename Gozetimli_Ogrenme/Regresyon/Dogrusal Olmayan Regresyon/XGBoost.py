from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

DM_train=xgb.DMatrix(data=X_train,label=y_train)
DM_test=xgb.DMatrix(data=X_test,label=y_test)
xgb_model=XGBRegressor().fit(X_train,y_train)

y_pred=xgb_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

xgb_grid={'colsample_bytree':[0.4,0.5,0.6,0.9,1],
          'n_estimators':[100,200,500,1000],
          'max_depth':[2,3,4,5,6],
          'learning_rate':[0.1,0.01,0.5]}
xgb=XGBRegressor()
xgb_cv=GridSearchCV(xgb,param_grid=xgb_grid,cv=10,versobe=2)
xgb_cv.fit(X_train,y_train)
print(xgb_cv.best_params_)
xgb_tuned=GradientBoostingRegressor(learning_rate=0.1,
                                    max_depth=2,
                                    n_estimators=1000,
                                    colsample_bytree=0.9)
xgb_tuned=xgb_tuned.fit(X_train,y_train)
y_pred=xgb_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))