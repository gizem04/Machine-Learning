from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
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

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
mlp_model=MLPRegressor(hidden_layer_sizes=(100,20)).fit(X_train_scaled,y_train)
print(mlp_model,mlp_model.n_layers_)

print(mlp_model.predict(X_train_scaled)[0:5])
y_pred=mlp_model.predict(X_test_scaled)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

mlp_params={'alpha':[0.1,0.01,0.02,0.005],
            'hidden_layer_sizes':[(20,20),(100,50,150),(300,200,150)],
            'activation':['relu','logistic']}
mlp_cv_model=GridSearchCV(mlp_model,mlp_params,cv=10)
mlp_cv_model.fit(X_train_scaled,y_train)
print(mlp_cv_model.best_params_)

mlp_tuned=MLPRegressor(alpha=0.02,hidden_layer_sizes=(10,50,150))
mlp_tuned.fit(X_train_scaled,y_train)
y_pred=mlp_tuned.predict((X_test_scaled))
print(np.sqrt(mean_squared_error(y_test,y_pred)))