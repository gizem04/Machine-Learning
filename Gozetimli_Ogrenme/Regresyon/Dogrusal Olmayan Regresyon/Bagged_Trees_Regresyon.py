from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score
from sklearn.ensemble import BaggingRegressor

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

bag_model=BaggingRegressor(bootstrap_features=True)
bag_model.fit(X_train,y_train)
#kac tane ağaç oluştuğunu görmek için
print(bag_model.n_estimators)
#ağaç özelliklerini görmek için
print(bag_model.estimators_)
#herbir ağactaki örnekleri görmek için
print(bag_model.estimators_samples_)
#herbir ağacın bağımsız değişkenlerine ulaşmak için
print(bag_model.estimators_features_)
#herbir modele özel değerlere erişmek için
print(bag_model.estimators_[0])

y_pred=bag_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
iki_y_pred=bag_model.estimators_[1].fit(X_train,y_train).predict(X_test)
print(np.sqrt(mean_squared_error(y_test,iki_y_pred)))
yedi_y_pred=bag_model.estimators_[4].fit(X_train,y_train).predict(X_test)
print(np.sqrt(mean_squared_error(y_test,yedi_y_pred)))

bag_params={"n_estimators":range(2,20)}
bag_cv_model=GridSearchCV(bag_model,bag_params,cv=10)
bag_cv_model.fit(X_train,y_train)
print(bag_cv_model.best_params_)
bag_tuned=BaggingRegressor(n_estimators=15,random_state=45)
bag_tuned.fit(X_train,y_train)
y_pred=bag_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))