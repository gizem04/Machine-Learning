import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np


ad=pd.read_csv("Advertising.csv",usecols=[1,2,3,4])
df=ad.copy()
print(df.head())
X=df.drop("sales",axis=1)
y=df["sales"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=40)
training=df.copy()

#statsmodels
lm=sm.OLS(y_train,X_train)
model=lm.fit()
print(model.summary())

#scikit-learn model
lm=LinearRegression()
model=lm.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)

#tahmin
yeni_veri=[[30],[10],[40]]
yeni_veri=pd.DataFrame(yeni_veri).T
print(model.predict(yeni_veri))
rmse=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))
rmse=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
model.score(X_train,y_train)
cross_val_r2_score=cross_val_score(model,X,y,cv=10,scoring="r2").mean()

