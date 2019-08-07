from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from skompiler import skompile
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
X_train=pd.DataFrame(X_train["Hits"])
X_test=pd.DataFrame(X_test["Hits"])

cart_model=DecisionTreeRegressor()
cart_model.fit(X_train,y_train)
X_grid=np.arange(min(np.array(X_train)),max(np.array(X_train)),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X_train,y_train,color='red')
plt.plot(X_grid,cart_model.predict(X_grid),color='blue')
plt.title('CART RESRESYON AĞACI')
plt.xlabel('atış sayısı(Hits)')
plt.ylabel('maaş(salary)')
plt.show()
print(skompile(cart_model.predict))
print(cart_model.predict(X_test)[0:5])
print(cart_model.predict([[91]]))
y_pred=cart_model.predict((X_test))
print(np.sqrt(mean_squared_error(y_test,y_pred)))
cart_params={"min_samples_split":range(2,100),
             "max_leaf_nodes":range(2,10)}
cart_cv_model=GridSearchCV(cart_model,cart_params,cv=10)
cart_cv_model.fit(X_train,y_train)
print(cart_cv_model.best_params_)
cart_tuned=DecisionTreeRegressor(max_leaf_nodes=9,min_samples_split=76)
cart_tuned.fit(X_train,y_train)
y_pred=cart_tuned.predict(X_test)
print("test hatası:",np.sqrt(mean_squared_error(y_test,y_pred)))

hit=pd.read_csv("Hitters.csv")
df=hit.copy()
df=df.dropna()
dms=pd.get_dummies(df[['League','Division','NewLeague']])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

cart_cv_model=GridSearchCV(cart_model,cart_params,cv=10)
cart_cv_model.fit(X_train,y_train)
print(cart_cv_model.best_params_)
cart_tuned=DecisionTreeRegressor(max_leaf_nodes=9,min_samples_split=37)
cart_tuned.fit(X_train,y_train)
y_pred=cart_tuned.predict(X_test)
print("test hatası:",np.sqrt(mean_squared_error(y_test,y_pred)))
