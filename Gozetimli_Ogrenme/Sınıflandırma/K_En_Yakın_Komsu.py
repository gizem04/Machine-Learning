from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

dia=pd.read_csv("10.1 diabetes.csv.csv")
df=dia.copy()
df=df.dropna()
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

knn=KNeighborsClassifier()
knn_model=knn.fit(X_train,y_train)
y_pred=knn_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

knn_params={"n_neighbors":np.arange(1,50)}
knn_cv=GridSearchCV(knn,knn_params,cv=10)
knn_cv.fit(X_train,y_train)
print("En iyi skor:"+str(knn_cv.best_score_))
print("En iyi parametreler:"+str(knn_cv.best_params_))

knn=KNeighborsClassifier(11)
knn_tuned=knn.fit(X_train,y_train)
print(knn_tuned.score(X_test,y_test))
y_pred=knn_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))