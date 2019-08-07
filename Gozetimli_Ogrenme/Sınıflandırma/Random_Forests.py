from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import *

dia=pd.read_csv("10.1 diabetes.csv.csv")
df=dia.copy()
df=df.dropna()
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)

rf_model=RandomForestClassifier().fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
print(accuracy_score(y_test,y_pred))
rf_params={"max_depth":[2,5,8,10],
           "max_features":[2,5,8],
           "n_estimators":[10,500,1000],
           "min_samples_split":[2,5,10]}
rf_model=RandomForestClassifier()
rf_cv_model=GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1,verbose=2)
rf_cv_model.fit(X_train,y_train)
print("En iyi paramereler:"+str(rf_cv_model.best_params_))
rf_tuned=RandomForestClassifier(max_depth=10,
                                max_features=5,
                                min_samples_split=10,
                                n_estimators=1000)
rf_tuned.fit(X_train,y_train)
y_pred=rf_tuned.predict(X_test)
print(accuracy_score(y_test,y_pred))
Importance=pd.DataFrame({"Importance":rf_tuned.feature_importances_*100},
                        index=X_train.columns)
Importance.sort_values(by="Importance",
                       axis=0,
                       ascending=True).plot(kind="barh",color="r")
plt.xlabel("Değişken Önem Düzeyleri")
plt.show()