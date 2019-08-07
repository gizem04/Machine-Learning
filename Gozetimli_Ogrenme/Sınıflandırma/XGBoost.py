from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import *

dia=pd.read_csv("10.1 diabetes.csv.csv")
df=dia.copy()
df=df.dropna()
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
#X=df["Pregnancies"]
X=pd.DataFrame(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

xgb_model=XGBClassifier().fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
xgb_params={'n_estimators':[100,500,1000,2000],
            'subsample':[0.6,0.8,1.0],
            'max_depth':[3,4,5,6],
            'learning_rate':[0.1,0.01,0.02,0.05],
            'min_samples_split':[2,5,10]}
xgb=XGBClassifier()
xgb_cv_model=GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2)
xgb_cv_model.fit(X_train,y_train)
print("En iyi paramereler:"+str(xgb_cv_model.best_params_))
xgb=XGBClassifier(learning_rate=0.01,
                  n_estimators=100,
                  max_depth=6,
                  min_samples_split=2,
                  subsample=0.8)
xgb_tuned=xgb.fit(X_train,y_train)
y_pred=xgb_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))

