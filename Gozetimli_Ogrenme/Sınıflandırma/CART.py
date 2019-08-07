from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from skompiler import skompile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.tree import DecisionTreeClassifier,tree
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

cart=DecisionTreeClassifier()
cart_model=cart.fit(X_train,y_train)
print(skompile(cart_model.predict))
y_pred=cart_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
cart_grid={"max_depth":range(1,10),
           "min_samples_split":range(2,50)}
cart=tree.DecisionTreeClassifier()
cart_cv=GridSearchCV(cart,cart_grid,cv=10,n_jobs=-1,verbose=2)
cart_cv_model=cart_cv.fit(X_train,y_train)
print("en iyi parametreler:"+str(cart_cv_model.best_params_))
cart=tree.DecisionTreeClassifier(max_depth=5,min_samples__split=19)
cart_tuned=cart.fit(X_train,y_train)
y_pred=cart_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))