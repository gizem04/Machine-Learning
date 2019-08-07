from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
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

gbm_model=GradientBoostingClassifier().fit(X_train,y_train)
y_pred=gbm_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
gbm_params={"learning_rate":[0.001,0.01,0.1,0.05],
            "n_estimators":[100,500,100],
            "max_depth":[3,5,10],
            "min_samples_split":[2,5,10]}
gbm=GradientBoostingClassifier()
gbm_cv=GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2)
gbm_cv.fit(X_train,y_train)
print("En iyi paramereler:"+str(gbm_cv.best_params_))
gbm=GradientBoostingClassifier(learning_rate=0.01,
                               n_estimators=500,
                               max_depth=5,
                               min_samples_split=2)
gbm_tuned=gbm.fit(X_train,y_train)
y_pred=gbm_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))
