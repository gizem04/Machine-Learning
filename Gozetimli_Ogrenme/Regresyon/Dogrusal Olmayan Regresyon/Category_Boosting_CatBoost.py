from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from catboost import CatBoostRegressor

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

catb=CatBoostRegressor()
catb_model=catb.fit(X_train,y_train)

y_pred=catb_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
catb_grid={'iterations':[200,500,1000,2000],
           'learning_rate':[0.01,0.03,0.05,0.1],
           'depth':[3,4,5,6,7,8]}
catb=CatBoostRegressor()
catb_cv_model=GridSearchCV(catb,catb_grid,cv=5,verbose=2)
catb_cv_model.fit(X_train,y_train)
print(catb_cv_model.best_params_)
catb_tuned=CatBoostRegressor(iterations=200,
                             learning_rate=0.01,
                             depth=8)
catb_tuned=catb_tuned.fit(X_train,y_train)
y_pred=catb_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
