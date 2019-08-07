from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from catboost import CatBoostClassifier
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

catb_model=CatBoostClassifier().fit(X_train,y_train)
y_pred=catb_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
catb_params={'max_depth':[3,8,5],
             'learning_rate':[0.1,0.01,0.05],
             'iterations':[200,500]}
catb=CatBoostClassifier()
catb_cv_model=GridSearchCV(catb,catb_params,cv=10,n_jobs=-1,verbose=2)
catb_cv_model.fit(X_train,y_train)
print("En iyi paramereler:"+str(catb_cv_model.best_params_))
catb=CatBoostClassifier(learning_rate=0.05,
                        iterations=200,
                        max_depth=5)
catb_tuned=catb.fit(X_train,y_train)
y_pred=catb_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))
