import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
training=df.copy()

enet_model=ElasticNet().fit(X_train,y_train)
y_pred=enet_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

enet_cv_model=ElasticNetCV(cv=10, random_state=0)
enet_cv_model.fit(X_train,y_train)
print(enet_cv_model.alpha_)
enet_tuned=ElasticNet(alpha=enet_cv_model.alpha_).fit(X_train,y_train)
y_pred=enet_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))