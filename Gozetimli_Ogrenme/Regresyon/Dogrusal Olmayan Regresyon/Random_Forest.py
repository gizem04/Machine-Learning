from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

rf_model=RandomForestRegressor(random_state=42)
rf_model.fit(X_train,y_train)

print(rf_model.predict(X_test)[0:5])
y_pred=rf_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

rf_params={'max_depth':list(range(1,10)),
           'max_features':[3,5,10,15],
           'n_estimators':[100,200,500,1000,2000]}
rf_cv_model=GridSearchCV(rf_model,rf_params,cv=10)
rf_cv_model.fit(X_train,y_train)
print(rf_cv_model.best_params_)
rf_tuned=RandomForestRegressor(max_depth=8,max_features=3,n_estimators=200)
rf_tuned.fit(X_train,y_train)
y_pred=rf_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
#değişkenlerin önem düzeyleri erişmek için
Importance=pd.DataFrame({"Importance":rf_tuned.feature_importances_*100},
                        index=X_train.columns)
Importance.sort_values(by="Importance",axis=0,ascending=True).plot(kind="barh",color="r")
plt.xlabel("değişken önem düzeyleri")
plt.show()