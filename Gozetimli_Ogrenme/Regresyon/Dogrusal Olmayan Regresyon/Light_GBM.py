from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from lightgbm import LGBMRegressor

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

lgbm=LGBMRegressor()
lgbm_model=lgbm.fit(X_train,y_train)
y_pred=lgbm_model.predict(X_test,num_iteration=best_iteration_)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
lgbm_grid={'learning_rate':[0.01,0.1,1,0.5],
           'colsample_bytree':[0.4,0.5,0.6,0.9,1],
           'max_depth':[3,5,8,50,100],
           'n_estimators':[20,40,100,200,500,1000]}
lgbm_cv_model=GridSearchCV(lgbm,lgbm_grid,cv=10,verbose=2)
lgbm_cv_model.fit(X_train,y_train)
print(lgbm_cv_model.best_params_)
lgbm_tuned=LGBMRegressor(learning_rate=0.1,
                         colsample_bytree=0.6,
                         max_depth=7,
                         n_estimators=40)
lgbm_tuned=lgbm_tuned.fit(X_train,y_train)
y_pred=lgbm_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
