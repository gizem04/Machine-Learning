import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

ad=pd.read_csv("Advertising.csv")
df=ad.copy()
df=df.iloc[:,1:len(df)]
df.head()
print(df.info())
print(df.describe().T)
print(df.isnull().values.any())
print(df.corr())
print(sns.pairplot(df,kind="reg"));
print(sns.jointplot(x="TV",y="sales",data=df,kind="reg"))

#statsmadels ile modelleme
X=df[["TV"]]
print(X[0:5])
X=sm.add_constant(X)
print(X[0:5])
y=df["sales"]
print(y[0:5])
lm=sm.OLS(y,X)
model=lm.fit()
print(model.summary())

#import statsmodels.formula.api as smf ile aynı modeli kurabiliriz.
lm=smf.ols("sales ~ TV",df)
model=lm.fit()
model.summary()

print(model.params)
print(model.summary().tables[1])
print(model.conf_int())
print("f_pvalue:","%.4f",model.f_pvalue)
print("fvalue:","%.4f",model.fvalue)
print("tvalue:","%.2f",model.tvalues[0:1])
print(model.mse_model)
print(model.rsquared)
print(model.rsquared_adj)
#buraya kadar modelimizi kurduk ve kendi kişielleştirimlerimize göre
#alabileceğimiz bazı şerleri aldık

#modelin tahmin ettiği değerlere erişmek
print(model.fittedvalues[0:5])
print(y[0:5])
print("sales="+ str("%.2f"%model.params[0])+"+TV"+"*"+str("%.2f"%model.params[1]))

#görsel olarak ifade etmek gerekirse
g=sns.regplot(df["TV"],df["sales"],ci=None,scatter_kws={'color':'r','s':9})
g.set_title("model demklemi:sales=7.03+TV*0.05")
g.set_ylabel("satış sayısı")
g.set_xlabel("TV harcamaları")
plt.xlim(-10,310);
plt.ylim(bottom=0);
plt.scatter(ad.TV,ad.sales)
plt.show()
