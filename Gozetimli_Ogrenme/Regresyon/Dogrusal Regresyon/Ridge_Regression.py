import pandas as pd
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
training=df.copy()

ridge_model=Ridge(alpha=0.1).fit(X_train,y_train)
lambdalar=10**np.linspace(10,-2,100)*0.5
ridge_model=Ridge()
katsayilar=[]
for i in lambdalar:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train,y_train)
    katsayilar.append(ridge_model.coef_)
"""ax=plt.gca()
ax=plot(lambdalar,katsayilar)
ax.set_xscale('log')
plt.xlabel('lambda(alpha) Değerleri')
plt.ylabel('Katsayılar/Ağırlıklar')
plt.title('Düzemleştirmenin bir fonksiyonu olarak Ridge Katsayıları');
plt.show()"""

y_pred=ridge_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
ridge_cv_model=RidgeCV(alphas=lambdalar,scoring='neg_mean_squared_error',normalize=True)
ridge_cv_model.fit(X_train,y_train)
print(ridge_cv_model.alpha_)
ridge_tuned=Ridge(alpha=ridge_cv_model.alpha_,normalize=True)
ridge_tuned.fit(X_train,y_train)
y_pred=ridge_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
