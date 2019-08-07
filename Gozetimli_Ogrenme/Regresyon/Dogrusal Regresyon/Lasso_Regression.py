import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
training=df.copy()

lasso_model=Lasso(alpha=0.1).fit(X_train,y_train)
lasso=Lasso()
lambdalar=10**np.linspace(10,-2,100)*0.5
katsayilar=[]
for i in lambdalar:
    lasso.set_params(alpha=i)
    lasso.fit(X_train,y_train)
    katsayilar.append(lasso.coef_)
"""ax=plt.gca()
ax.plot(lambdalar*2,katsayilar)
ax.set_scale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()"""

y_pred=lasso_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
lasso_cv_model=LassoCV(alphas=None,cv=10, max_iter=10000,normalize=True)
lasso_cv_model.fit(X_train,y_train)
print(lasso_cv_model.alpha_)
lasso_tuned=Lasso(alpha=lasso_cv_model.alpha_)
lasso_tuned.fit(X_train,y_train)
y_pred=lasso_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
