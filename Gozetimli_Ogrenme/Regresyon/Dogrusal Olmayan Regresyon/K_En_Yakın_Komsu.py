import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
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

knn_model=KNeighborsRegressor().fit(X_train,y_train)
print(knn_model.n_neighbors)
y_pred=knn_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

RMSE=[]
RMSE_CV=[]
for k in range(10):
    k=k+1
    knn_model=KNeighborsRegressor(n_neighbors=k).fit(X_train,y_train)
    y_pred=knn_model.predict(X_train)
    rmse=np.sqrt(mean_squared_error(y_train,y_pred))
    rmse_cv=np.sqrt(-1*cross_val_score(knn_model,X_train,y_train,cv=10,scoring="neg_mean_squared_error").mean())
    RMSE.append(rmse)
    RMSE_CV.append(rmse_cv)
    print("k={} için RMSE değeri: {}".format(k,RMSE))
    print("k={} için RMSE değeri: {}".format(k, RMSE_CV))

knn_params={'n_neighbors':np.arange(1,30,1)}
print(np.arange(1,30,1))
knn=KNeighborsRegressor()
knn_cv_model=GridSearchCV(knn,knn_params,cv=10)
print(knn_cv_model)
knn_cv_model.fit(X_train,y_train)
print(knn_cv_model.best_params_["n_neighbors"])

knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(X_train,y_train)
print(np.sqrt(mean_squared_error(y_test,knn_tuned.predict(X_test))))
