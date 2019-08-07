from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


dia=pd.read_csv("10.1 diabetes.csv.csv")
df=dia.copy()
df=df.dropna()
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

nb=GaussianNB()
nb_model=nb.fit(X_train,y_train)
print(nb_model.predict(X_test)[0:10])
print(nb_model.predict_proba(X_test)[0:10])
print(accuracy_score(y_test,nb_model.predict(X_test)))
print(cross_val_score(nb_model,X_test,y_test,cv=10).mean())
