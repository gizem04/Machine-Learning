from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  statsmodels.api as sm
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


dia=pd.read_csv("10.1 diabetes.csv.csv")
df=dia.copy()
df=df.dropna()
print(df.info())
print(df["Outcome"].value_counts())
df["Outcome"].value_counts().plot.barh();
print(df.describe().T)

#statsmodel
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)
loj=sm.Logit(y,X)
loj_model=loj.fit()
print(loj_model.summary())

#scikit-learn
loj=LogisticRegression(solver="liblinear")
loj_model=loj.fit(X,y)
print(loj_model.intercept_)
print(loj_model.coef_)

y_pred=loj_model.predict(X)
print(confusion_matrix(y,y_pred))
#accuracy_score--->doğru yaatığımız olaylar/tüm olaylar
print(accuracy_score(y,y_pred))
print(classification_report(y,y_pred))
print(loj_model.predict(X)[0:10])
print(loj_model.predict_proba(X)[0:10])
print(y[0:10])
y_probs=loj_model.predict_proba(X)
y_probs=y_probs[:,1]
y_pred=[1 if i>0.5 else 0 for i in y_probs]
print(y_pred[0:10])
print(confusion_matrix(y,y_pred))
print(accuracy_score(y,y_pred))
print(classification_report(y,y_pred))
logit_roc_auc=roc_auc_score(y,loj_model.predict(X))
fpr,tpr,thresholds=roc_curve(y,loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr,tpr,label='AUC(area=%0.2f)'%logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
loj=LogisticRegression(solver="liblinear")
loj_model=loj.fit(X_train,y_train)
print(accuracy_score(y_test,loj_model.predict(X_test)))
print(cross_val_score(loj_model,X_test,y_test,cv=10))
print(cross_val_score(loj_model,X_test,y_test,cv=10).mean())