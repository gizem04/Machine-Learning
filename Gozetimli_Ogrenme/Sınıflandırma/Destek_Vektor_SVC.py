from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.metrics import *
import matplotlib.pyplot as plt

dia=pd.read_csv("10.1 diabetes.csv.csv")
df=dia.copy()
df=df.dropna()
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

svm_model=SVC(kernel="linear").fit(X_train,y_train)
y_pred=svm_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
svc_params={"C":np.arange(1,10)}
svc=SVC(kernel="linear")
svc_cv_model=GridSearchCV(svc,svc_params,cv=10,n_jobs=-1,verbose=2)
svc_cv_model.fit(X_train, y_train)
print("En iyi parametreler:"+str(svc_cv_model.best_params_))
svc_tuned=SVC(kernel="linear",C=5).fit(X_train,y_train)
y_pred=svc_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))

