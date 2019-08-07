import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



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

svr_model=SVR("linear").fit(X_train,y_train)
print(svr_model.predict(X_train)[0:5])
print("y={0}+{1}x".format(svr_model.intercept_[0], svr_model.coef_[0][0]))
print(X_train["Hits"][0:1])
print((svr_model.intercept_[0])+(svr_model.coef_[0][0])*(X_train["Hits"][0:1]))
y_pred=svr_model.predict(X_train)
plt.scatter(X_train,y_train)
plt.plot(X_train,y_pred,color='r')
plt.show()

lm_model=LinearRegression().fit(X_train,y_train)
lm_pred=lm_model.predict(X_train)
print("y={0}+{1}x".format(lm_model.intercept_, lm_model.coef_[0]))
print((lm_model.intercept_)+(lm_model.coef_[0])*(X_train["Hits"][0:1]))
plt.scatter(X_train,y_train,alpha=0.5,s=23)
plt.plot(X_train,lm_pred,'g')
plt.plot(X_train,y_pred,'r')
plt.xlabel("atış sayısı(Hits)")
plt.ylabel("maaş(salary)")
plt.show()

y_pred=svr_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
svr_params={"C":np.arange(0.1,2,0.1)}
svr_cv_model=GridSearchCV(svr_model,svr_params,cv=10).fit(X_train,y_train)
print(pd.Series(svr_cv_model.best_params_)[0])
svr_tuned=SVR("linear",C=pd.Series(svr_cv_model.best_params_)[0]).fit(X_train,y_train)
y_pred=svr_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
