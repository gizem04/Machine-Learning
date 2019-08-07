import pandas as pd
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression,PLSSVD
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error
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

pls_model=PLSRegression().fit(X_train,y_train)
y_pred=pls_model.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,y_pred)))
y_pred=pls_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
cv_10=model_selection.KFold(n_splits=10,shuffle=True,random_state=1)
RMSE=[]
for i in np.arange(1,X_train.shape[1]+1):
    pls=PLSRegression(n_components=i)
    score=np.sqrt(-1*cross_val_score(pls,X_train,y_train,cv=cv_10,scoring='neg_mean_squared_error').mean())
    RMSE.append(score)
plt.plot(np.arange(1,X_train.shape[1]+1),np.array(RMSE),'-v',c="r")
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Salary')
plt.show()

pls_model=PLSRegression(n_components=2).fit(X_train,y_train)
y_pred=pls_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
