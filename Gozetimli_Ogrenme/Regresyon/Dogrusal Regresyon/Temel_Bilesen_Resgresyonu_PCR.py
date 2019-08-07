import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
df.head()
df.info
print(df.describe().T)
dms=pd.get_dummies(df[['League','Division','NewLeague']])
dms.head()
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=40)
training=df.copy()

pca=PCA()
X_reduced_train=pca.fit_transform(scale(X_train))
np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)[0:10]
lm=LinearRegression()
pcr_model=lm.fit(X_reduced_train,y_train)
print(pcr_model.intercept_)
print(pcr_model.coef_)
y_pred=pcr_model.predict(X_reduced_train)
print(y_pred[0:5])
np.sqrt(mean_squared_error(y_train,y_pred))

pca2=PCA()
X_reduced_test=pca2.fit_transform(scale(X_test))
y_pred=pcr_model.predict(X_reduced_test)
np.sqrt(mean_squared_error(y_test,y_pred))
cv_10=model_selection.KFold(n_splits=10,shuffle=True,random_state=1)
RMSE=[]
for i in np.arange(1,X_reduced_train.shape[1]+1):
    score=np.sqrt(-1*model_selection.cross_val_score(lm,
                                                     X_reduced_train[:,:i],
                                                     y_train.ravel(),
                                                     cv=cv_10,
                                                     scoring='neg_mean_squared_error').mean())
    RMSE.append(score)
plt.plot(RMSE,'-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Maas Tahmin Modeli için PCR Model Tuning')
plt.show()

pcr_model=lm.fit(X_reduced_train[:,0:16],y_train)
y_pred=pcr_model.predict(X_reduced_train[:,0:16])
print(np.sqrt(mean_squared_error(y_train,y_pred)))
