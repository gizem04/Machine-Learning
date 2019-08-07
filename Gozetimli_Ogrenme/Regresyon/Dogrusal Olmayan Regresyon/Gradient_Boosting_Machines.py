from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

gbm_model=GradientBoostingRegressor()
gbm_model.fit(X_train,y_train)

y_pred=gbm_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

gbm_params={'learning_rate':[0.001,0.01,0.1,0.2],
            'max_depth':[3,5,8,50,100],
            'n_estimators':[200,500,1000,2000],
            'subsample':[1,0.5,0.75]}
gbm=GradientBoostingRegressor()
gbm_cv_model=GridSearchCV(gbm,gbm_params,cv=10,verbose=2)
gbm_cv_model.fit(X_train,y_train)
print(gbm_cv_model.best_params_)
gbm_tuned=GradientBoostingRegressor(learning_rate=0.1,
                                    max_depth=3,
                                    n_estimators=200,
                                    subsample=0.5)
gbm_tuned=gbm_tuned.fit(X_train,y_train)
y_pred=gbm_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
#değişken önem düzeyleri
Importance=pd.DataFrame({"Importance":gbm_tuned.feature_importances_*100},
                        index=X_train.columns)
Importance.sort_values(by="Importance",axis=0,ascending=True).plot(kind="barh",color="r")
plt.xlabel("değişken önem düzeyleri")
plt.show()